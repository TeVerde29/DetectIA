package unu.edu.detectia

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import kotlin.math.max
import kotlin.math.roundToInt

/**
 * Resultado del preprocesamiento estilo MNIST.
 */
sealed class PreprocessResult {
    data class Success(
        val features: FloatArray,
        val previewBitmap: Bitmap
    ) : PreprocessResult()

    data object EmptyCanvas : PreprocessResult()
}

/**
 * Replica el preprocesamiento oficial de MNIST:
 *  1. Detecta el bounding box de los píxeles blancos dibujados.
 *  2. Recorta el dígito y lo escala manteniendo proporción para que su lado mayor mida 20 px.
 *  3. Lo centra por centro de masa dentro de un lienzo negro de 28x28 px.
 *  4. Normaliza los 784 píxeles en [0.0f, 1.0f] (valor / 255.0f).
 */
object MnistPreprocessor {

    private const val TARGET_BOX_SIZE = 20
    private const val MNIST_SIZE = 28
    private const val PIXEL_THRESHOLD = 20

    fun process(sourceBitmap: Bitmap): PreprocessResult {
        val width = sourceBitmap.width
        val height = sourceBitmap.height
        if (width <= 0 || height <= 0) return PreprocessResult.EmptyCanvas

        val pixels = IntArray(width * height)
        sourceBitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        // 1. Encontrar el bounding box de los píxeles dibujados
        var minX = width
        var minY = height
        var maxX = -1
        var maxY = -1

        for (y in 0 until height) {
            val rowOffset = y * width
            for (x in 0 until width) {
                val intensity = extractIntensity(pixels[rowOffset + x])
                if (intensity > PIXEL_THRESHOLD) {
                    if (x < minX) minX = x
                    if (x > maxX) maxX = x
                    if (y < minY) minY = y
                    if (y > maxY) maxY = y
                }
            }
        }

        if (maxX < minX || maxY < minY) {
            return PreprocessResult.EmptyCanvas
        }

        val cropWidth = maxX - minX + 1
        val cropHeight = maxY - minY + 1

        // 2. Recortar y redimensionar manteniendo proporción (lado mayor = 20 px)
        val cropped = Bitmap.createBitmap(sourceBitmap, minX, minY, cropWidth, cropHeight)
        val targetWidth: Int
        val targetHeight: Int
        if (cropWidth >= cropHeight) {
            targetWidth = TARGET_BOX_SIZE
            targetHeight = max(1, (TARGET_BOX_SIZE.toFloat() * cropHeight / cropWidth).roundToInt())
        } else {
            targetHeight = TARGET_BOX_SIZE
            targetWidth = max(1, (TARGET_BOX_SIZE.toFloat() * cropWidth / cropHeight).roundToInt())
        }

        val scaled = Bitmap.createScaledBitmap(cropped, targetWidth, targetHeight, true)
        if (cropped !== sourceBitmap) {
            cropped.recycle()
        }

        // 3. Calcular centro de masa sobre la imagen escalada 20px
        val scaledPixels = IntArray(targetWidth * targetHeight)
        scaled.getPixels(scaledPixels, 0, targetWidth, 0, 0, targetWidth, targetHeight)

        var massSum = 0.0
        var weightedX = 0.0
        var weightedY = 0.0

        for (y in 0 until targetHeight) {
            val rowOffset = y * targetWidth
            for (x in 0 until targetWidth) {
                val intensity = extractIntensity(scaledPixels[rowOffset + x]).toDouble()
                if (intensity > 0.0) {
                    massSum += intensity
                    weightedX += x * intensity
                    weightedY += y * intensity
                }
            }
        }

        if (massSum <= 0.0) {
            scaled.recycle()
            return PreprocessResult.EmptyCanvas
        }

        val comX = weightedX / massSum
        val comY = weightedY / massSum

        // Centrar el centro de masa en el centro geométrico del lienzo 28x28 (13.5, 13.5)
        val canvasCenter = (MNIST_SIZE - 1) / 2.0
        val maxOffsetX = MNIST_SIZE - targetWidth
        val maxOffsetY = MNIST_SIZE - targetHeight
        val offsetX = (canvasCenter - comX).roundToInt().coerceIn(0, maxOffsetX)
        val offsetY = (canvasCenter - comY).roundToInt().coerceIn(0, maxOffsetY)

        val mnistBitmap = Bitmap.createBitmap(MNIST_SIZE, MNIST_SIZE, Bitmap.Config.ARGB_8888)
        val mnistCanvas = Canvas(mnistBitmap)
        mnistCanvas.drawColor(Color.BLACK)
        val drawPaint = Paint(Paint.FILTER_BITMAP_FLAG)
        mnistCanvas.drawBitmap(scaled, offsetX.toFloat(), offsetY.toFloat(), drawPaint)
        scaled.recycle()

        // 4. Extraer los 784 píxeles en orden fila-mayor y normalizar a [0.0f, 1.0f]
        val finalPixels = IntArray(MNIST_SIZE * MNIST_SIZE)
        mnistBitmap.getPixels(finalPixels, 0, MNIST_SIZE, 0, 0, MNIST_SIZE, MNIST_SIZE)

        val features = FloatArray(MNIST_SIZE * MNIST_SIZE)
        for (i in finalPixels.indices) {
            val intensity = extractIntensity(finalPixels[i])
            features[i] = intensity / 255.0f
            // Asegura vista previa puramente en escala de grises
            finalPixels[i] = Color.rgb(intensity, intensity, intensity)
        }
        mnistBitmap.setPixels(finalPixels, 0, MNIST_SIZE, 0, 0, MNIST_SIZE, MNIST_SIZE)

        return PreprocessResult.Success(
            features = features,
            previewBitmap = mnistBitmap
        )
    }

    private fun extractIntensity(argb: Int): Int {
        val r = (argb ushr 16) and 0xFF
        val g = (argb ushr 8) and 0xFF
        val b = argb and 0xFF
        return max(r, max(g, b))
    }
}
