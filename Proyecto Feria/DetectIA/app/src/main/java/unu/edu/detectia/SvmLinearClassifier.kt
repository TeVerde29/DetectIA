package unu.edu.detectia

import android.content.Context
import org.json.JSONObject
import kotlin.math.abs
import kotlin.math.exp

/**
 * Probabilidad y detalle de puntuación para un dígito individual.
 */
data class DigitProbability(
    val digit: Int,
    val probabilityPercent: Float,
    val votes: Int,
    val continuousScore: Double
)

/**
 * Resultado completo de la inferencia SVM Lineal One-vs-One.
 */
data class SvmPredictionResult(
    val predictedDigit: Int,
    val confidencePercent: Float,
    val inferenceTimeMs: Double,
    val probabilities: List<DigitProbability>,
    val votesByDigit: IntArray
)

/**
 * Motor de inferencia SVM Lineal 100% nativo en Kotlin (sin librerías externas).
 * Implementa la función de decisión One-vs-One de scikit-learn para 10 clases
 * (45 clasificadores binarios w · x + b) y convierte las puntuaciones continuas
 * mediante Softmax en porcentajes de confianza.
 */
class SvmLinearClassifier private constructor(
    private val inputSize: Int,
    private val numClassifiers: Int,
    private val classes: IntArray,
    private val intercepts: DoubleArray,
    private val coefFlat: FloatArray
) {

    private val numClasses: Int = classes.size

    fun predict(features: FloatArray): SvmPredictionResult {
        require(features.size == inputSize) {
            "Se esperaban $inputSize características, pero se recibieron ${features.size}."
        }

        val startNs = System.nanoTime()

        val votes = IntArray(numClasses)
        val sumConfidences = DoubleArray(numClasses)

        // 1. Evaluar los 45 hiperplanos One-vs-One: d_k = w_k · x + b_k
        var classifierIndex = 0
        for (i in 0 until numClasses) {
            for (j in i + 1 until numClasses) {
                val rowOffset = classifierIndex * inputSize
                var dot = 0.0
                for (f in 0 until inputSize) {
                    dot += coefFlat[rowOffset + f] * features[f]
                }
                val decision = dot + intercepts[classifierIndex]

                if (decision > 0.0) {
                    votes[i]++
                } else {
                    votes[j]++
                }

                sumConfidences[i] += decision
                sumConfidences[j] -= decision
                classifierIndex++
            }
        }

        // 2. Puntuación continua por dígito (votos + margen acotado estilo scikit-learn)
        val continuousScores = DoubleArray(numClasses)
        var maxScore = Double.NEGATIVE_INFINITY
        for (c in 0 until numClasses) {
            val sumConf = sumConfidences[c]
            val transformedConf = sumConf / (3.0 * (abs(sumConf) + 1.0))
            val score = votes[c].toDouble() + transformedConf
            continuousScores[c] = score
            if (score > maxScore) {
                maxScore = score
            }
        }

        // 3. Softmax estable numéricamente para obtener porcentajes (0% a 100%)
        val expScores = DoubleArray(numClasses)
        var expSum = 0.0
        for (c in 0 until numClasses) {
            val expVal = exp(continuousScores[c] - maxScore)
            expScores[c] = expVal
            expSum += expVal
        }

        val probabilities = ArrayList<DigitProbability>(numClasses)
        for (c in 0 until numClasses) {
            val percent = ((expScores[c] / expSum) * 100.0).toFloat()
            probabilities.add(
                DigitProbability(
                    digit = classes[c],
                    probabilityPercent = percent,
                    votes = votes[c],
                    continuousScore = continuousScores[c]
                )
            )
        }

        probabilities.sortByDescending { it.probabilityPercent }
        val topResult = probabilities.first()
        val elapsedMs = (System.nanoTime() - startNs) / 1_000_000.0

        return SvmPredictionResult(
            predictedDigit = topResult.digit,
            confidencePercent = topResult.probabilityPercent,
            inferenceTimeMs = elapsedMs,
            probabilities = probabilities,
            votesByDigit = votes
        )
    }

    companion object {
        private const val DEFAULT_ASSET_NAME = "modelo_linear_android.json"

        fun loadFromAssets(
            context: Context,
            assetFileName: String = DEFAULT_ASSET_NAME
        ): SvmLinearClassifier {
            val jsonText = context.assets.open(assetFileName).bufferedReader().use {
                it.readText()
            }
            val root = JSONObject(jsonText)

            val inputSize = root.getInt("input_size")
            val numClassifiers = root.getInt("num_classifiers")

            val classesArray = root.getJSONArray("classes")
            val classes = IntArray(classesArray.length()) { idx ->
                classesArray.getInt(idx)
            }

            val interceptArray = root.getJSONArray("intercept")
            val intercepts = DoubleArray(interceptArray.length()) { idx ->
                interceptArray.getDouble(idx)
            }

            val coefMatrix = root.getJSONArray("coef")
            val expectedRows = classes.size * (classes.size - 1) / 2
            require(coefMatrix.length() == numClassifiers && numClassifiers == expectedRows) {
                "Cantidad de clasificadores inconsistente en el modelo JSON."
            }

            val coefFlat = FloatArray(numClassifiers * inputSize)
            for (row in 0 until numClassifiers) {
                val rowArray = coefMatrix.getJSONArray(row)
                val baseOffset = row * inputSize
                for (col in 0 until inputSize) {
                    coefFlat[baseOffset + col] = rowArray.getDouble(col).toFloat()
                }
            }

            return SvmLinearClassifier(
                inputSize = inputSize,
                numClassifiers = numClassifiers,
                classes = classes,
                intercepts = intercepts,
                coefFlat = coefFlat
            )
        }
    }
}
