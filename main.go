//go:build js && wasm
// +build js,wasm

package main

import (
	"bytes"
	"encoding/base64"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"image/jpeg"
	"image/png"
	"math"
	"sort"
	"strings"
	"syscall/js"
	"time"
)

// const marginSize = 50

// Point は2D座標を表します
type Point struct {
	X, Y int
}

// Contour は画像上の輪郭を表します
type Contour []Point

// 画像処理に必要な構造体
type ImageProcessor struct {
	Width  int
	Height int
}

// Wasmのエクスポート関数
func main() {
	// WebAssemblyの場合はこちらの関数が呼ばれる
	c := make(chan struct{})

	// JavaScriptのグローバルオブジェクトを取得
	js.Global().Set("detectAndCropReceipts", js.FuncOf(detectAndCropReceipts))

	fmt.Println("Go WebAssembly initialized. Ready to process images.")
	<-c // プログラムが終了しないようにチャネルをブロック
}

// JavaScript側から呼び出される関数
func detectAndCropReceipts(this js.Value, args []js.Value) interface{} {
	if len(args) < 1 {
		return js.ValueOf("No image data provided")
	}
	fmt.Println("Processing image...")

	// Base64エンコードされた画像を取得
	imageData := args[0].String()

	// "data:image/png;base64," などのプレフィックスを取り除く
	prefixIndex := strings.Index(imageData, ",")
	if prefixIndex < 0 {
		return js.ValueOf("Invalid image data format")
	}

	// Base64デコード
	imageBytes, err := base64.StdEncoding.DecodeString(imageData[prefixIndex+1:])
	if err != nil {
		fmt.Println("Base64 decode error:", err)
		return js.ValueOf("Error decoding image data")
	}

	// 画像をデコード
	var img image.Image
	var decodeErr error

	// PNGかJPEGかを判定して適切にデコード
	if bytes.HasPrefix(imageBytes, []byte{0x89, 'P', 'N', 'G'}) {
		img, decodeErr = png.Decode(bytes.NewReader(imageBytes))
	} else if bytes.HasPrefix(imageBytes, []byte{0xFF, 0xD8}) {
		img, decodeErr = jpeg.Decode(bytes.NewReader(imageBytes))
	} else {
		// その他のフォーマットも試す
		img, _, decodeErr = image.Decode(bytes.NewReader(imageBytes))
	}

	if decodeErr != nil {
		fmt.Println("Image decode error:", decodeErr)
		return js.ValueOf("Error processing image")
	}

	bounds := img.Bounds()
	width, height := bounds.Dx(), bounds.Dy()
	marginSize := min(width, height) / 20
	// 新しい画像サイズを計算（マージン込み）
	newWidth := width + 2*marginSize
	newHeight := height + 2*marginSize

	// 新しい画像を作成
	newImg := image.NewRGBA(image.Rect(0, 0, newWidth, newHeight))
	draw.Draw(newImg, newImg.Bounds(), image.NewUniform(color.White), image.Point{}, draw.Src)

	// 元の画像を新しい画像の中央に配置
	draw.Draw(newImg, image.Rect(marginSize, marginSize, marginSize+width, marginSize+height),
		img, bounds.Min, draw.Src)

	// レシートを検出して切り取る
	croppedImages := processReceiptImage(newImg)

	// 切り取った画像をBase64でエンコードして返す
	resultArray := js.Global().Get("Array").New(len(croppedImages))
	for i, croppedImg := range croppedImages {
		// 画像をエンコード
		var buf bytes.Buffer
		err := png.Encode(&buf, croppedImg)
		if err != nil {
			fmt.Println("Error encoding result image:", err)
			continue
		}

		// Base64エンコード
		base64Data := "data:image/png;base64," + base64.StdEncoding.EncodeToString(buf.Bytes())
		resultArray.SetIndex(i, base64Data)
	}

	return resultArray
}

// 画像処理のメイン関数
func processReceiptImage(img image.Image) []image.Image {
	// 2値化
	binaryImg := binarize(img)

	// ノイズ除去
	binaryImg = noiseReduction(binaryImg)

	// 輪郭検出
	contours := findContours(binaryImg)
	fmt.Printf("検出された輪郭の数: %d\n", len(contours))

	// 矩形の検出
	approxContours := approximateContours(img, contours)
	fmt.Printf("検出された矩形の数: %d\n", len(approxContours))

	// 切り取った画像を保存
	return cropReceiptImages(img, approxContours)
}

// 検出された矩形部分を切り取り、輪郭線も描画する
func cropReceiptImages(img image.Image, contours []Contour) []image.Image {
	// 元画像の境界を取得
	bounds := img.Bounds()

	// 結果画像の配列
	var croppedImages []image.Image

	for _, contour := range contours {
		// バウンディングボックスを計算
		minX, maxX, minY, maxY := findBoundingBox(contour)

		// 少し余白を追加
		padding := 5
		minX = max(0, minX-padding)
		minY = max(0, minY-padding)
		maxX = min(bounds.Max.X, maxX+padding)
		maxY = min(bounds.Max.Y, maxY+padding)

		// 切り取り範囲の矩形を作成
		rect := image.Rect(minX, minY, maxX, maxY)

		// 元画像から部分を切り取る
		croppedImg := image.NewRGBA(image.Rect(0, 0, rect.Dx(), rect.Dy()))
		draw.Draw(croppedImg, croppedImg.Bounds(), img, rect.Min, draw.Src)

		// 輪郭線を描画
		drawContourOnImage(croppedImg, contour, rect.Min, color.RGBA{R: 255, G: 0, B: 0, A: 255}) // 赤色で輪郭を描画

		// 結果配列に追加
		croppedImages = append(croppedImages, croppedImg)
	}

	return croppedImages
}

// 画像に輪郭線を描画し、面積を計算して表示する
func drawContourOnImage(img *image.RGBA, contour Contour, offset image.Point, lineColor color.Color) {
	if len(contour) < 2 {
		return
	}
	n := len(contour)

	// 輪郭の点を順番に線で結ぶ
	for i := 0; i < n; i++ {
		j := (i + 1) % n
		// 線を引く（ブレゼンハムのアルゴリズム）
		drawLine(img,
			contour[i].X-offset.X, contour[i].Y-offset.Y,
			contour[j].X-offset.X, contour[j].Y-offset.Y,
			lineColor)
	}

	// オフセット情報をデバッグ出力
	fmt.Printf("オフセット情報: X=%d, Y=%d\n", offset.X, offset.Y)

	// 元の座標系での面積を計算
	originalArea := calculatePolygonArea(contour)

	// 面積の計算に使用した各点の座標をデバッグ出力
	if n <= 10 { // 点数が多すぎる場合は出力を制限
		fmt.Println("輪郭の座標点:")
		for i, p := range contour {
			fmt.Printf("  点%d: (%d, %d)\n", i, p.X, p.Y)
		}
	}

	// 面積を画像の左上に表示
	drawAreaText(img, originalArea)
}

// 多角形の面積を計算する (Shoelace formula)
func calculatePolygonArea(contour Contour) float64 {
	n := len(contour)
	if n < 3 {
		return 0
	}

	// バウンディングボックスを計算して面積を確認（デバッグ用）
	minX, maxX, minY, maxY := findBoundingBox(contour)
	width := maxX - minX
	height := maxY - minY
	rectArea := width * height

	// デバッグ情報
	fmt.Printf("輪郭点数: %d, バウンディングボックス: %d x %d = %d 平方ピクセル\n",
		n, width, height, rectArea)

	// Shoelace formula (ガウスの面積公式)で計算
	area := 0.0
	for i := 0; i < n; i++ {
		j := (i + 1) % n
		area += float64(contour[i].X*contour[j].Y - contour[j].X*contour[i].Y)
	}

	polygonArea := math.Abs(area) / 2.0
	fmt.Printf("計算された多角形面積: %.2f 平方ピクセル\n", polygonArea)

	// 面積が極端に小さい場合はバウンディングボックスの面積を返す
	// if polygonArea < float64(rectArea)*0.1 {
	// 	fmt.Println("警告: 計算された面積が小さすぎるため、バウンディングボックス面積を使用")
	// 	return float64(rectArea)
	// }

	return polygonArea
}

// 画像に面積のテキストを描画する
func drawAreaText(img *image.RGBA, area float64) {
	// 画像の左上に面積を表示
	bounds := img.Bounds()
	x := bounds.Min.X + 5
	y := bounds.Min.Y + 15

	// 面積を文字列に変換
	areaText := fmt.Sprintf("Area: %.1f", area)

	// 背景を描画（読みやすくするため）
	for dy := -1; dy <= 1; dy++ {
		for dx := -1; dx <= 1; dx++ {
			drawSimpleText(img, areaText, x+dx, y+dy, color.RGBA{R: 0, G: 0, B: 0, A: 255})
		}
	}

	// テキストを描画
	drawSimpleText(img, areaText, x, y, color.RGBA{R: 0, G: 255, B: 0, A: 255})
}

// シンプルなテキスト描画
func drawSimpleText(img *image.RGBA, text string, x, y int, textColor color.Color) {
	// ピクセルごとに文字を描画する単純な実装
	// 実際のアプリケーションでは、フォントライブラリを使用することを推奨
	posX := x

	// 各文字分のスペースを確保して点を打つ
	for i := 0; i < len(text); i++ {
		// 文字の幅（簡易表示のため固定値）
		charWidth := 8

		// 点を打つだけの単純な表示
		img.Set(posX, y, textColor)

		// 次の文字位置へ
		posX += charWidth
	}
}

// ブレゼンハムのアルゴリズムで線を描画
func drawLine(img *image.RGBA, x0, y0, x1, y1 int, clr color.Color) {
	dx := abs(x1 - x0)
	dy := abs(y1 - y0)
	sx, sy := 1, 1
	if x0 >= x1 {
		sx = -1
	}
	if y0 >= y1 {
		sy = -1
	}
	err := dx - dy

	bounds := img.Bounds()
	for {
		// 画像の範囲内かチェック
		if x0 >= bounds.Min.X && x0 < bounds.Max.X && y0 >= bounds.Min.Y && y0 < bounds.Max.Y {
			img.Set(x0, y0, clr)
		}

		if x0 == x1 && y0 == y1 {
			break
		}
		e2 := 2 * err
		if e2 > -dy {
			err -= dy
			x0 += sx
		}
		if e2 < dx {
			err += dx
			y0 += sy
		}
	}
}

// 画像を2値化する（ガウシアン法 - 高精度マルチパスアプローチ）
func binarize(img image.Image) *image.Gray {
	// 処理時間計測
	start := time.Now()
	defer func() {
		fmt.Printf("二値化処理時間: %v\n", time.Since(start))
	}()

	bounds := img.Bounds()
	width, height := bounds.Dx(), bounds.Dy()
	grayImg := image.NewGray(bounds)
	binaryImg := image.NewGray(bounds)

	// グレースケール変換
	pixels := make([]uint8, width*height)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r, g, b, _ := img.At(x+bounds.Min.X, y+bounds.Min.Y).RGBA()
			gray := uint8((0.299*float64(r) + 0.587*float64(g) + 0.114*float64(b)) / 256.0)
			idx := y*width + x
			pixels[idx] = gray
			grayImg.SetGray(x+bounds.Min.X, y+bounds.Min.Y, color.Gray{Y: gray})
		}
	}

	fmt.Printf("グレースケール変換完了: %v\n", time.Since(start))

	blockSize := 255
	c := 2
	halfBlock := blockSize / 2
	thresholdDist := halfBlock * halfBlock

	// 改善点1: 精度向上のための事前フィルタリング
	// ガウシアン円形カーネルを近似する重みマトリックスを生成
	weights := make([][]float64, blockSize)
	for i := range weights {
		weights[i] = make([]float64, blockSize)
	}

	// より精密な円形カーネルの重みを計算
	totalWeight := 0.0
	for y := 0; y < blockSize; y++ {
		dy := y - halfBlock
		for x := 0; x < blockSize; x++ {
			dx := x - halfBlock
			dist := dx*dx + dy*dy
			if dist <= thresholdDist {
				// 精度を向上させるため、距離に基づいた重みを設定
				// 中心に近いほど重みが大きくなる
				weight := 1.0 - math.Sqrt(float64(dist))/float64(halfBlock)
				weights[y][x] = weight
				totalWeight += weight
			}
		}
	}

	// 重みの正規化
	for y := 0; y < blockSize; y++ {
		for x := 0; x < blockSize; x++ {
			if weights[y][x] > 0 {
				weights[y][x] /= totalWeight
			}
		}
	}

	fmt.Printf("重み計算完了: %v\n", time.Since(start))

	// 改善点2: 正確な水平方向カーネルと垂直方向カーネルの分解
	// 特異値分解（SVD）の近似を使用して2Dカーネルを1D水平・垂直カーネルに分解
	// 計算を簡略化するため近似的な方法を使用

	// 水平カーネル（x方向）
	hKernel := make([]float64, blockSize)
	for x := 0; x < blockSize; x++ {
		sum := 0.0
		for y := 0; y < blockSize; y++ {
			sum += weights[y][x]
		}
		hKernel[x] = math.Sqrt(sum)
	}

	// 垂直カーネル（y方向）
	vKernel := make([]float64, blockSize)
	for y := 0; y < blockSize; y++ {
		sum := 0.0
		for x := 0; x < blockSize; x++ {
			sum += weights[y][x]
		}
		vKernel[y] = math.Sqrt(sum)
	}

	// カーネルの正規化
	hSum, vSum := 0.0, 0.0
	for i := 0; i < blockSize; i++ {
		hSum += hKernel[i]
		vSum += vKernel[i]
	}
	for i := 0; i < blockSize; i++ {
		hKernel[i] /= hSum
		vKernel[i] /= vSum
	}

	// 改善点3: 浮動小数点数を使って精度を向上
	// 中間結果を浮動小数点数で保持
	hSumFloat := make([]float64, width*height)
	hCountFloat := make([]float64, width*height)

	// 水平方向の畳み込み（第1パス）
	fmt.Printf("水平方向畳み込み開始: %v\n", time.Since(start))
	for y := 0; y < height; y++ {
		baseIdx := y * width
		for x := 0; x < width; x++ {
			sum := 0.0
			weightSum := 0.0

			// 各ピクセルの水平方向のカーネル適用
			for kx := 0; kx < blockSize; kx++ {
				dx := kx - halfBlock
				nx := x + dx

				// 画像の境界チェック
				if nx >= 0 && nx < width {
					// カーネルの重みを適用
					kernelWeight := hKernel[kx]
					if kernelWeight > 0 {
						sum += float64(pixels[baseIdx+nx]) * kernelWeight
						weightSum += kernelWeight
					}
				}
			}

			// 正規化
			if weightSum > 0 {
				hSumFloat[baseIdx+x] = sum / weightSum * weightSum
				hCountFloat[baseIdx+x] = weightSum
			}
		}
	}

	// 垂直方向の畳み込み（第2パス）
	fmt.Printf("垂直方向畳み込み開始: %v\n", time.Since(start))
	finalSumFloat := make([]float64, width*height)
	finalCountFloat := make([]float64, width*height)

	for x := 0; x < width; x++ {
		for y := 0; y < height; y++ {
			sum := 0.0
			weightSum := 0.0

			// 各ピクセルの垂直方向のカーネル適用
			for ky := 0; ky < blockSize; ky++ {
				dy := ky - halfBlock
				ny := y + dy

				// 画像の境界チェック
				if ny >= 0 && ny < height {
					// カーネルの重みを適用
					kernelWeight := vKernel[ky]
					if kernelWeight > 0 {
						weight := hCountFloat[ny*width+x]
						if weight > 0 {
							sum += hSumFloat[ny*width+x] * kernelWeight
							weightSum += weight * kernelWeight
						}
					}
				}
			}

			// 正規化
			if weightSum > 0 {
				finalSumFloat[y*width+x] = sum
				finalCountFloat[y*width+x] = weightSum
			}
		}
	}

	// 改善点4: エッジの保存と詳細の強調
	// エッジ検出のための勾配計算（ソーベルフィルタの簡易版）
	fmt.Printf("エッジ検出開始: %v\n", time.Since(start))
	edgeMap := make([]float64, width*height)
	maxGradient := 0.0

	for y := 1; y < height-1; y++ {
		for x := 1; x < width-1; x++ {
			// 水平と垂直の勾配計算
			gx := float64(pixels[y*width+(x+1)]) - float64(pixels[y*width+(x-1)])
			gy := float64(pixels[(y+1)*width+x]) - float64(pixels[(y-1)*width+x])

			// 勾配の大きさ
			gradient := math.Sqrt(gx*gx + gy*gy)
			edgeMap[y*width+x] = gradient

			if gradient > maxGradient {
				maxGradient = gradient
			}
		}
	}

	// 勾配マップの正規化
	if maxGradient > 0 {
		for i := range edgeMap {
			edgeMap[i] /= maxGradient
		}
	}

	// 最終的な二値化処理（エッジを考慮）
	fmt.Printf("二値化処理開始: %v\n", time.Since(start))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			idx := y*width + x

			count := finalCountFloat[idx]
			if count <= 0 {
				continue
			}

			// 平均値の計算
			meanValue := finalSumFloat[idx] / count

			// エッジに基づく閾値の調整（エッジでは閾値を下げて詳細を保存）
			edgeFactor := 1.0 - edgeMap[idx]*0.5 // エッジでは最大50%閾値を下げる

			// 閾値 = 平均値 - C (エッジファクターで調整)
			threshold := uint8(meanValue) - uint8(float64(c)*edgeFactor)
			pixelValue := pixels[idx]

			// 2値化
			var value uint8
			if pixelValue > threshold {
				value = 255
			}

			binaryImg.SetGray(x+bounds.Min.X, y+bounds.Min.Y, color.Gray{Y: value})
		}
	}

	fmt.Printf("処理完了: %v\n", time.Since(start))
	return binaryImg
}

// ノイズ除去（中央値フィルタ）
func noiseReduction(img *image.Gray) *image.Gray {
	bounds := img.Bounds()
	width, height := bounds.Dx(), bounds.Dy()
	result := image.NewGray(bounds)

	// 中央値フィルタのサイズ
	filterSize := 9
	halfFilter := filterSize / 2

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			// 近傍のピクセル値を収集
			values := make([]uint8, 0, filterSize*filterSize)

			for ny := max(0, y-halfFilter); ny < min(height, y+halfFilter+1); ny++ {
				for nx := max(0, x-halfFilter); nx < min(width, x+halfFilter+1); nx++ {
					values = append(values, img.GrayAt(nx+bounds.Min.X, ny+bounds.Min.Y).Y)
				}
			}

			// 中央値を求める
			sort.Slice(values, func(i, j int) bool {
				return values[i] < values[j]
			})

			medianValue := values[len(values)/2]
			result.SetGray(x+bounds.Min.X, y+bounds.Min.Y, color.Gray{Y: medianValue})
		}
	}

	return result
}

// 輪郭検出（改良版 - 境界追跡アルゴリズム）
func findContours(img *image.Gray) []Contour {
	bounds := img.Bounds()
	width, height := bounds.Dx(), bounds.Dy()

	// 処理用の画像をコピー
	processImg := image.NewGray(bounds)
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			processImg.SetGray(x+bounds.Min.X, y+bounds.Min.Y, img.GrayAt(x+bounds.Min.X, y+bounds.Min.Y))
		}
	}

	// エッジ検出（Sobel演算子）を適用
	edgeImg := detectEdges(processImg)

	// エッジの二値化と細線化
	thresholdedImg := thresholdEdges(edgeImg, 50)

	// 輪郭を保存するスライス
	var contours []Contour

	// 訪問済みピクセルを追跡
	visited := make([][]bool, height)
	for i := range visited {
		visited[i] = make([]bool, width)
	}

	// 外部輪郭のみを追跡するため、画像の境界から1ピクセル内側をスキャン
	for y := 1; y < height-1; y++ {
		for x := 1; x < width-1; x++ {
			// 輪郭点（黒ピクセル）かつ未訪問の場合
			if thresholdedImg.GrayAt(x+bounds.Min.X, y+bounds.Min.Y).Y == 0 && !visited[y][x] {
				// 境界追跡アルゴリズムで輪郭を抽出
				contour := traceContour(thresholdedImg, x, y, visited)

				// 意味のある輪郭のみを保持（短すぎる輪郭は除外）
				if len(contour) > 20 {
					// 直線的な輪郭のみを保持する
					// if isLinearContour(contour) && !isTouchingImageBorder(contour, width, height) {
					contours = append(contours, contour)
					// }
				}
			}
		}
	}

	fmt.Printf("検出された輪郭の数: %d\n", len(contours))
	return contours
}

// エッジ検出（Sobel演算子）
func detectEdges(img *image.Gray) *image.Gray {
	bounds := img.Bounds()
	width, height := bounds.Dx(), bounds.Dy()
	result := image.NewGray(bounds)

	// Sobelカーネル
	sobelX := [][]int{
		{-1, 0, 1},
		{-2, 0, 2},
		{-1, 0, 1},
	}

	sobelY := [][]int{
		{-1, -2, -1},
		{0, 0, 0},
		{1, 2, 1},
	}

	// 各ピクセルに対してSobel演算を適用
	for y := 1; y < height-1; y++ {
		for x := 1; x < width-1; x++ {
			// X方向とY方向の勾配を計算
			gradX := 0
			gradY := 0

			for ky := -1; ky <= 1; ky++ {
				for kx := -1; kx <= 1; kx++ {
					pixel := int(img.GrayAt(x+kx+bounds.Min.X, y+ky+bounds.Min.Y).Y)
					gradX += pixel * sobelX[ky+1][kx+1]
					gradY += pixel * sobelY[ky+1][kx+1]
				}
			}

			// 勾配の大きさを計算
			magnitude := uint8(math.Min(255, math.Sqrt(float64(gradX*gradX+gradY*gradY))))
			result.SetGray(x+bounds.Min.X, y+bounds.Min.Y, color.Gray{Y: magnitude})
		}
	}

	return result
}

// エッジの二値化
func thresholdEdges(img *image.Gray, threshold uint8) *image.Gray {
	bounds := img.Bounds()
	width, height := bounds.Dx(), bounds.Dy()
	result := image.NewGray(bounds)

	// 二値化
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			if img.GrayAt(x+bounds.Min.X, y+bounds.Min.Y).Y > threshold {
				result.SetGray(x+bounds.Min.X, y+bounds.Min.Y, color.Gray{Y: 0}) // エッジ
			} else {
				result.SetGray(x+bounds.Min.X, y+bounds.Min.Y, color.Gray{Y: 255}) // 背景
			}
		}
	}

	return result
}

// 境界追跡アルゴリズム
func traceContour(img *image.Gray, startX, startY int, visited [][]bool) Contour {
	bounds := img.Bounds()
	width, height := bounds.Dx(), bounds.Dy()

	// 画像の輪郭も検出するため、開始点が画像の端に近い場合は画像全体の矩形を返す
	borderThreshold := 2
	if startX <= borderThreshold || startX >= width-borderThreshold ||
		startY <= borderThreshold || startY >= height-borderThreshold {
		// 画像の四隅を結ぶ矩形を作成
		rect := Contour{
			{X: 0, Y: 0},                  // 左上
			{X: width - 1, Y: 0},          // 右上
			{X: width - 1, Y: height - 1}, // 右下
			{X: 0, Y: height - 1},         // 左下
			{X: 0, Y: 0},                  // 左上（閉じる）
		}

		// 訪問済みにマーク（画像の縁全体）
		for y := 0; y < height; y++ {
			if y == 0 || y == height-1 {
				// 上端と下端の行は全部マーク
				for x := 0; x < width; x++ {
					visited[y][x] = true
				}
			} else {
				// 中間の行は左端と右端だけマーク
				visited[y][0] = true
				visited[y][width-1] = true
			}
		}

		return rect
	}

	contour := Contour{}
	current := Point{X: startX, Y: startY}
	contour = append(contour, current)
	visited[startY][startX] = true

	// 8方向の近傍を定義（時計回り）
	directions := []Point{
		{X: 1, Y: 0},   // 右
		{X: 1, Y: 1},   // 右下
		{X: 0, Y: 1},   // 下
		{X: -1, Y: 1},  // 左下
		{X: -1, Y: 0},  // 左
		{X: -1, Y: -1}, // 左上
		{X: 0, Y: -1},  // 上
		{X: 1, Y: -1},  // 右上
	}

	// 輪郭追跡の開始方向
	dirIdx := 0
	startPoint := current

	// 輪郭が閉じるまで追跡を続ける
	for {
		found := false
		// 8方向を調べる（最大8ステップ）
		for i := 0; i < 8; i++ {
			// 現在の方向から順に調べる
			newDirIdx := (dirIdx + i) % 8
			newDir := directions[newDirIdx]

			newX, newY := current.X+newDir.X, current.Y+newDir.Y

			// 画像の範囲内チェック
			inBounds := newX >= 0 && newX < width && newY >= 0 && newY < height

			// 範囲外の場合または黒いピクセルの場合、エッジとみなす
			isEdge := false
			if !inBounds {
				// 画像の範囲外はエッジとみなす
				isEdge = true
				// 範囲外の点を画像の境界に補正
				newX = max(0, min(newX, width-1))
				newY = max(0, min(newY, height-1))
			} else if img.GrayAt(newX+bounds.Min.X, newY+bounds.Min.Y).Y == 0 {
				// 黒いピクセルはエッジ
				isEdge = true
			}

			// エッジ点であれば処理
			if isEdge && !visited[newY][newX] {
				visited[newY][newX] = true
				current = Point{X: newX, Y: newY}
				contour = append(contour, current)

				// 次の探索方向を更新
				dirIdx = (newDirIdx + 4) % 8
				found = true
				break
			}
		}

		// 次の点が見つからない場合、またはスタート地点に戻った場合は終了
		if !found || (len(contour) > 2 && current.X == startPoint.X && current.Y == startPoint.Y) {
			break
		}
	}

	return contour
}

// 輪郭を近似して矩形を検出
func approximateContours(img image.Image, contours []Contour) []Contour {
	bounds := img.Bounds()
	width, height := bounds.Dx(), bounds.Dy()
	imgSize := width * height

	var approxContours []Contour

	for i, contour := range contours {
		// 周囲長を計算
		arcLength := calculateArcLength(contour, true)

		// 面積を計算
		// area := calculateContourArea(contour, width, height)
		area := calculatePolygonArea(contour)
		// area := calculateAreaUsingPick(contour)
		maxArea := calculateContourMaxArea(contour, width, height)
		// 条件に合う輪郭のみ処理
		marginSize := min(width, height) / 20
		for _, p := range contour {
			if p.X < marginSize || p.X >= width-marginSize || p.Y < marginSize || p.Y >= height-marginSize {
				area = 0
				break
			}
		}
		if arcLength != 0 && area > float64(imgSize)*0.02 && area < float64(imgSize)*0.9 {
			fmt.Printf("Contour %d: 点数=%d, 面積=%.2f (画像の%.2f%%), (画像の%.2f%%), 周囲長=%.2f\n",
				i, len(contour), area, (area/float64(imgSize))*100, (maxArea/float64(imgSize))*100, arcLength)
			// 輪郭を近似
			approxContour := approxPolyDP(contour, 0.05*arcLength, true)
			// 凸包を適用して凸多角形を保証する
			convexContour := findConvexHull(approxContour)

			approxContours = append(approxContours, convexContour)
		}
	}

	return approxContours
}

// 輪郭のバウンディングボックスを見つける
func findBoundingBox(contour Contour) (minX, maxX, minY, maxY int) {
	if len(contour) == 0 {
		return 0, 0, 0, 0
	}

	minX, maxX = contour[0].X, contour[0].X
	minY, maxY = contour[0].Y, contour[0].Y

	for _, p := range contour {
		if p.X < minX {
			minX = p.X
		}
		if p.X > maxX {
			maxX = p.X
		}
		if p.Y < minY {
			minY = p.Y
		}
		if p.Y > maxY {
			maxY = p.Y
		}
	}

	return minX, maxX, minY, maxY
}

// 凸包を計算する（Graham's Scanアルゴリズム）
func findConvexHull(points Contour) Contour {
	n := len(points)
	if n <= 3 {
		return points // 3点以下は既に凸包
	}

	// ステップ1: Y座標が最も小さい点（複数ある場合はX座標が最も小さい点）を見つける
	lowestPoint := 0
	for i := 1; i < n; i++ {
		if points[i].Y < points[lowestPoint].Y ||
			(points[i].Y == points[lowestPoint].Y && points[i].X < points[lowestPoint].X) {
			lowestPoint = i
		}
	}

	// 基準点を最初の位置に移動
	if lowestPoint != 0 {
		points[0], points[lowestPoint] = points[lowestPoint], points[0]
	}

	// コピーを作成（ソート操作のため）
	sortPoints := make(Contour, n)
	copy(sortPoints, points)

	// ステップ2: 残りの点を極角でソート
	// 基準点
	p0 := sortPoints[0]

	// カスタムソート関数（極角に基づく）
	sort.Slice(sortPoints[1:], func(i, j int) bool {
		// インデックスの調整（points[1:]の中でのインデックスi, j）
		i, j = i+1, j+1

		// 極角を比較（三角形の面積の符号で決定）
		orientation := crossProduct(p0, sortPoints[i], sortPoints[j])

		if orientation == 0 {
			// 同一直線上の場合は距離で比較
			distI := distanceSquared(p0, sortPoints[i])
			distJ := distanceSquared(p0, sortPoints[j])
			return distI < distJ
		}

		// 反時計回りの順序
		return orientation > 0
	})

	// ステップ3: Graham's Scanで凸包を構築
	hull := make(Contour, 0, n)
	hull = append(hull, sortPoints[0], sortPoints[1])

	for i := 2; i < n; i++ {
		// 右折している限り、スタックから点を取り除く
		for len(hull) > 1 && crossProduct(hull[len(hull)-2], hull[len(hull)-1], sortPoints[i]) <= 0 {
			hull = hull[:len(hull)-1]
		}
		hull = append(hull, sortPoints[i])
	}

	return hull
}

// 三点の外積（反時計回りなら正、時計回りなら負、一直線上なら0）
func crossProduct(p0, p1, p2 Point) int {
	return (p1.X-p0.X)*(p2.Y-p0.Y) - (p2.X-p0.X)*(p1.Y-p0.Y)
}

// 二点間の距離の二乗
func distanceSquared(p1, p2 Point) int {
	dx := p1.X - p2.X
	dy := p1.Y - p2.Y
	return dx*dx + dy*dy
}

// 周囲長を計算
func calculateArcLength(contour Contour, closed bool) float64 {
	if len(contour) < 2 {
		return 0
	}

	length := 0.0
	for i := 0; i < len(contour)-1; i++ {
		dx := float64(contour[i+1].X - contour[i].X)
		dy := float64(contour[i+1].Y - contour[i].Y)
		length += math.Sqrt(dx*dx + dy*dy)
	}

	if closed && len(contour) > 0 {
		dx := float64(contour[0].X - contour[len(contour)-1].X)
		dy := float64(contour[0].Y - contour[len(contour)-1].Y)
		length += math.Sqrt(dx*dx + dy*dy)
	}

	return length
}

// 最大矩形の面積を計算
func calculateContourMaxArea(contour Contour, imageWidth, imageHeight int) float64 {
	minX, minY := imageWidth, imageHeight
	maxX, maxY := 0, 0

	for _, pt := range contour {
		if pt.X < minX {
			minX = pt.X
		}
		if pt.X > maxX {
			maxX = pt.X
		}
		if pt.Y < minY {
			minY = pt.Y
		}
		if pt.Y > maxY {
			maxY = pt.Y
		}
	}
	return float64((maxX - minX + 1) * (maxY - minY + 1))
}

// 最大矩形の面積を計算
func calculateContourArea(contour Contour, imageWidth, imageHeight int) float64 {
	n := len(contour)
	area := 0.0
	for i := 0; i < n; i++ {
		j := (i + 1) % n
		area += float64(contour[i].X*contour[j].Y - contour[j].X*contour[i].Y)
	}

	return math.Abs(area) / 2.0
}

// 輪郭の面積を計算 (Shoelace formula)
func calculateContourArea1(contour Contour) float64 {
	n := len(contour)
	if n < 3 {
		return 0
	}

	// 画像全体の矩形かどうかをチェック（5点で、最初と最後が同じ場合）
	if n == 5 && contour[0].X == contour[4].X && contour[0].Y == contour[4].Y {
		// 矩形の特徴をチェック
		if contour[0].X == 0 && contour[0].Y == 0 && // 左上
			contour[1].X > 0 && contour[1].Y == 0 && // 右上
			contour[2].X > 0 && contour[2].Y > 0 && // 右下
			contour[3].X == 0 && contour[3].Y > 0 { // 左下
			// 画像全体を表す矩形の場合、単純に面積を計算
			width := contour[1].X + 1  // 0からの座標なので+1
			height := contour[3].Y + 1 // 0からの座標なので+1
			return float64(width * height)
		}
	}

	// 画像の境界に接している点があるかチェック
	hasBoundaryPoints := false
	var maxX, maxY int = 0, 0
	for _, p := range contour {
		if p.X > maxX {
			maxX = p.X
		}
		if p.Y > maxY {
			maxY = p.Y
		}

		if p.X == 0 || p.Y == 0 { // 画像の左側または上側の境界
			hasBoundaryPoints = true
		}
	}

	// 画像の境界に接している場合、輪郭を画像の境界線で補完
	if hasBoundaryPoints {
		// コピーを作成して変更
		closedContour := make(Contour, len(contour))
		copy(closedContour, contour)

		// 最初と最後の点が同じでない場合（閉じていない場合）
		if len(contour) > 0 && (contour[0].X != contour[len(contour)-1].X || contour[0].Y != contour[len(contour)-1].Y) {
			closedContour = append(closedContour, contour[0]) // 閉じる
		}

		// 標準的なShoelace formulaで面積を計算
		area := 0.0
		for i := 0; i < len(closedContour)-1; i++ {
			j := i + 1
			area += float64(closedContour[i].X*closedContour[j].Y - closedContour[j].X*closedContour[i].Y)
		}

		return math.Abs(area) / 2.0
	}

	// 通常の輪郭の場合はShoelace formula
	area := 0.0
	for i := 0; i < n; i++ {
		j := (i + 1) % n
		area += float64(contour[i].X*contour[j].Y - contour[j].X*contour[i].Y)
	}

	return math.Abs(area) / 2.0
}

// 輪郭の面積を計算 (Modified Shoelace formula for border contours)
func calculateContourArea2(contour Contour, imageWidth, imageHeight int) float64 {
	n := len(contour)
	if n < 3 {
		return 0
	}

	// Check if contour touches image borders
	touchesBorder := false
	for _, pt := range contour {
		if pt.X == 0 || pt.X == imageWidth-1 || pt.Y == 0 || pt.Y == imageHeight-1 {
			touchesBorder = true
			break
		}
	}

	// Standard shoelace formula calculation
	area := 0.0
	for i := 0; i < n; i++ {
		j := (i + 1) % n
		area += float64(contour[i].X*contour[j].Y - contour[j].X*contour[i].Y)
	}

	// If the contour touches the image border, add the area of a bounding rectangle
	if touchesBorder {
		// Find bounding box
		minX, minY := imageWidth, imageHeight
		maxX, maxY := 0, 0

		for _, pt := range contour {
			if pt.X < minX {
				minX = pt.X
			}
			if pt.X > maxX {
				maxX = pt.X
			}
			if pt.Y < minY {
				minY = pt.Y
			}
			if pt.Y > maxY {
				maxY = pt.Y
			}
		}

		// If contour touches left/right borders, extend X
		if minX == 0 {
			minX = 0
		}
		if maxX == imageWidth-1 {
			maxX = imageWidth - 1
		}

		// If contour touches top/bottom borders, extend Y
		if minY == 0 {
			minY = 0
		}
		if maxY == imageHeight-1 {
			maxY = imageHeight - 1
		}

		// Calculate bounding box area
		boundingBoxArea := float64((maxX - minX + 1) * (maxY - minY + 1))

		// Use a weighted combination
		// You may need to adjust this weighting factor based on your specific needs
		return math.Max(math.Abs(area)/2.0, boundingBoxArea*0.8)
	}

	return math.Abs(area) / 2.0
}

// 多角形近似（Ramer-Douglas-Peucker アルゴリズム）- 高速化版
func approxPolyDP(contour Contour, epsilon float64, closed bool) Contour {
	n := len(contour)
	if n <= 2 {
		return contour
	}

	// 結果を格納するスライスを事前に確保（最悪の場合、全ての点を保持）
	result := make(Contour, 0, n)

	// 反復処理用のスタック
	type segment struct {
		start, end int
	}
	stack := make([]segment, 0, n)

	// 初期セグメントをスタックに追加
	if closed {
		// 閉じた輪郭の場合、最も遠い頂点を見つける（O(n)で最適化）
		farthestIdx := 0
		maxDist := 0.0

		// 基準点として最初の点を使用
		basePoint := contour[0]

		// 一度のループで最も遠い点を見つける
		for i := 1; i < n; i++ {
			dx := float64(contour[i].X - basePoint.X)
			dy := float64(contour[i].Y - basePoint.Y)
			dist := dx*dx + dy*dy // ルートは不要（比較のみなので）

			if dist > maxDist {
				maxDist = dist
				farthestIdx = i
			}
		}

		// 最も遠い点と元の点でセグメントを作成
		stack = append(stack, segment{0, farthestIdx})
		stack = append(stack, segment{farthestIdx, n - 1})

		// 閉じた輪郭の場合、最後の点から最初の点へのセグメントも追加
		if farthestIdx != 0 && farthestIdx != n-1 {
			stack = append(stack, segment{n - 1, 0})
		}
	} else {
		// 開いた輪郭の場合は単純に最初と最後の点でセグメント作成
		stack = append(stack, segment{0, n - 1})
	}

	// セグメントの処理済みフラグ（インデックスの重複処理を防止）
	processed := make([]bool, n)

	// スタックが空になるまで処理を続ける
	for len(stack) > 0 {
		// スタックからセグメントを取り出す
		lastIdx := len(stack) - 1
		seg := stack[lastIdx]
		stack = stack[:lastIdx]

		start, end := seg.start, seg.end

		// セグメントが短すぎる場合は処理しない
		if abs(end-start) <= 1 {
			if !processed[start] {
				result = append(result, contour[start])
				processed[start] = true
			}
			if !processed[end] {
				result = append(result, contour[end])
				processed[end] = true
			}
			continue
		}

		// 最大距離とそのインデックスを見つける
		maxDist := 0.0
		maxIdx := start

		// 線分の方程式 ax + by + c = 0 の係数を計算
		startPoint := contour[start]
		endPoint := contour[end]
		a := float64(endPoint.Y - startPoint.Y)
		b := float64(startPoint.X - endPoint.X)
		c := float64(startPoint.Y*endPoint.X - startPoint.X*endPoint.Y)

		// 正規化係数（線分の長さ）- 一度だけ計算
		norm := math.Sqrt(a*a + b*b)
		if norm < 1e-10 { // 0除算を防止
			norm = 1.0
		}

		// 各点と線分の距離を計算
		for i := start + 1; i < end; i++ {
			p := contour[i]
			// 点と線分の距離（前もって係数計算済み）
			dist := math.Abs(a*float64(p.X)+b*float64(p.Y)+c) / norm
			if dist > maxDist {
				maxDist = dist
				maxIdx = i
			}
		}

		// 最大距離がepsilonより大きい場合、さらに分割
		if maxDist > epsilon {
			stack = append(stack, segment{start, maxIdx})
			stack = append(stack, segment{maxIdx, end})
		} else {
			// epsilon以下なら、startとendの点のみ保持
			if !processed[start] {
				result = append(result, contour[start])
				processed[start] = true
			}
			if !processed[end] {
				result = append(result, contour[end])
				processed[end] = true
			}
		}
	}

	return result
}

// ヘルパー関数
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}
