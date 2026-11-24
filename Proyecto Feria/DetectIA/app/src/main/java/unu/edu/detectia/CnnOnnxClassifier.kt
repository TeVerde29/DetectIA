package unu.edu.detectia

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.nio.FloatBuffer
import kotlin.math.exp

/**
 * Probabilidad Softmax para un dígito individual.
 */
data class DigitProbability(
    val digit: Int,
    val probabilityPercent: Float
)

/**
 * Resultado de inferencia de la Red Neuronal Convolucional (ONNX).
 */
data class CnnPredictionResult(
    val predictedDigit: Int,
    val confidencePercent: Float,
    val inferenceTimeMs: Double,
    val probabilities: List<DigitProbability>
)

/**
 * Motor de inferencia CNN 100% offline usando ONNX Runtime para Android.
 * Recibe un vector de 784 píxeles normalizados en [0.0f, 1.0f], construye
 * un tensor (1, 1, 28, 28) y aplica Softmax a los 10 logits de salida.
 */
class CnnOnnxClassifier private constructor(
    private val env: OrtEnvironment,
    private val session: OrtSession,
    private val inputName: String
) : AutoCloseable {

    fun predict(features: FloatArray): CnnPredictionResult {
        require(features.size == INPUT_SIZE) {
            "Se esperaban $INPUT_SIZE características, pero se recibieron ${features.size}."
        }

        val startNs = System.nanoTime()

        val shape = longArrayOf(1L, 1L, 28L, 28L)
        val buffer = FloatBuffer.wrap(features)

        val logits: FloatArray = OnnxTensor.createTensor(env, buffer, shape).use { inputTensor ->
            session.run(mapOf(inputName to inputTensor)).use { results ->
                @Suppress("UNCHECKED_CAST")
                val output = results[0].value as Array<FloatArray>
                output[0]
            }
        }

        val numClasses = logits.size
        var maxLogit = Float.NEGATIVE_INFINITY
        for (v in logits) {
            if (v > maxLogit) maxLogit = v
        }

        val expVals = DoubleArray(numClasses)
        var expSum = 0.0
        for (i in 0 until numClasses) {
            val e = exp((logits[i] - maxLogit).toDouble())
            expVals[i] = e
            expSum += e
        }

        val probabilities = ArrayList<DigitProbability>(numClasses)
        for (i in 0 until numClasses) {
            val percent = ((expVals[i] / expSum) * 100.0).toFloat()
            probabilities.add(
                DigitProbability(
                    digit = i,
                    probabilityPercent = percent
                )
            )
        }

        probabilities.sortByDescending { it.probabilityPercent }
        val topResult = probabilities.first()
        val elapsedMs = (System.nanoTime() - startNs) / 1_000_000.0

        return CnnPredictionResult(
            predictedDigit = topResult.digit,
            confidencePercent = topResult.probabilityPercent,
            inferenceTimeMs = elapsedMs,
            probabilities = probabilities
        )
    }

    override fun close() {
        session.close()
    }

    companion object {
        private const val INPUT_SIZE = 784
        private const val DEFAULT_ASSET_NAME = "modelo_cnn.onnx"

        fun loadFromAssets(
            context: Context,
            assetFileName: String = DEFAULT_ASSET_NAME
        ): CnnOnnxClassifier {
            val env = OrtEnvironment.getEnvironment()
            val modelBytes = context.assets.open(assetFileName).use { it.readBytes() }
            val session = env.createSession(modelBytes)
            val inputName = session.inputNames.first()
            return CnnOnnxClassifier(
                env = env,
                session = session,
                inputName = inputName
            )
        }
    }
}
