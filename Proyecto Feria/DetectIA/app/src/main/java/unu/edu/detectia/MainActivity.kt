package unu.edu.detectia

import android.content.res.Configuration
import android.graphics.Bitmap
import android.os.Bundle
import android.view.View
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.enableEdgeToEdge
import androidx.appcompat.app.AppCompatActivity
import androidx.appcompat.app.AppCompatDelegate
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import com.google.android.material.button.MaterialButton
import com.google.android.material.progressindicator.LinearProgressIndicator
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlin.math.roundToInt

class MainActivity : AppCompatActivity() {

    private lateinit var drawingView: DrawingView
    private lateinit var tvCanvasHint: TextView
    private lateinit var btnThemeToggle: MaterialButton
    private lateinit var btnClear: MaterialButton
    private lateinit var btnPredict: MaterialButton

    private lateinit var tvPredictedDigit: TextView
    private lateinit var tvConfidence: TextView
    private lateinit var tvLatency: TextView
    private lateinit var ivPreview28: ImageView

    private lateinit var topRows: List<TopRowViews>

    private val backgroundExecutor: ExecutorService = Executors.newSingleThreadExecutor()

    @Volatile
    private var classifier: SvmLinearClassifier? = null
    private var currentPreviewBitmap: Bitmap? = null

    private data class TopRowViews(
        val rank: Int,
        val label: TextView,
        val progress: LinearProgressIndicator,
        val percent: TextView
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        val prefs = getSharedPreferences("detectia_settings", MODE_PRIVATE)
        val savedMode = prefs.getInt("theme_mode", AppCompatDelegate.MODE_NIGHT_NO)
        if (AppCompatDelegate.getDefaultNightMode() != savedMode) {
            AppCompatDelegate.setDefaultNightMode(savedMode)
        }

        setContentView(R.layout.activity_main)

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main)) { v, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom)
            insets
        }

        bindViews()
        updateThemeToggleUi(isNightModeActive())
        setupListeners()
        loadSvmModelAsync()
    }

    private fun bindViews() {
        drawingView = findViewById(R.id.drawingView)
        tvCanvasHint = findViewById(R.id.tvCanvasHint)
        btnThemeToggle = findViewById(R.id.btnThemeToggle)
        btnClear = findViewById(R.id.btnClear)
        btnPredict = findViewById(R.id.btnPredict)

        tvPredictedDigit = findViewById(R.id.tvPredictedDigit)
        tvConfidence = findViewById(R.id.tvConfidence)
        tvLatency = findViewById(R.id.tvLatency)
        ivPreview28 = findViewById(R.id.ivPreview28)

        topRows = listOf(
            TopRowViews(
                rank = 1,
                label = findViewById(R.id.tvTop1Label),
                progress = findViewById(R.id.pbTop1),
                percent = findViewById(R.id.tvTop1Percent)
            ),
            TopRowViews(
                rank = 2,
                label = findViewById(R.id.tvTop2Label),
                progress = findViewById(R.id.pbTop2),
                percent = findViewById(R.id.tvTop2Percent)
            ),
            TopRowViews(
                rank = 3,
                label = findViewById(R.id.tvTop3Label),
                progress = findViewById(R.id.pbTop3),
                percent = findViewById(R.id.tvTop3Percent)
            )
        )
    }

    private fun setupListeners() {
        btnThemeToggle.setOnClickListener {
            toggleTheme()
        }

        drawingView.onDrawStateChanged = { hasContent ->
            tvCanvasHint.visibility = if (hasContent) View.GONE else View.VISIBLE
        }

        btnClear.setOnClickListener {
            drawingView.clear()
            resetResults()
        }

        btnPredict.setOnClickListener {
            runPrediction()
        }
    }

    private fun isNightModeActive(): Boolean {
        val defaultNight = AppCompatDelegate.getDefaultNightMode()
        return if (defaultNight == AppCompatDelegate.MODE_NIGHT_YES) {
            true
        } else if (defaultNight == AppCompatDelegate.MODE_NIGHT_NO) {
            false
        } else {
            (resources.configuration.uiMode and Configuration.UI_MODE_NIGHT_MASK) == Configuration.UI_MODE_NIGHT_YES
        }
    }

    private fun updateThemeToggleUi(isNight: Boolean) {
        btnThemeToggle.setIconResource(if (isNight) R.drawable.ic_mode_light else R.drawable.ic_mode_night)
        btnThemeToggle.text = if (isNight) "Claro" else "Oscuro"
        drawingView.setNightMode(isNight)
        WindowCompat.getInsetsController(window, window.decorView).isAppearanceLightStatusBars = !isNight
    }

    private fun toggleTheme() {
        val isNight = isNightModeActive()
        val newMode = if (isNight) AppCompatDelegate.MODE_NIGHT_NO else AppCompatDelegate.MODE_NIGHT_YES
        val prefs = getSharedPreferences("detectia_settings", MODE_PRIVATE)
        prefs.edit().putInt("theme_mode", newMode).apply()
        AppCompatDelegate.setDefaultNightMode(newMode)
    }

    private fun loadSvmModelAsync() {
        backgroundExecutor.execute {
            try {
                val loaded = SvmLinearClassifier.loadFromAssets(applicationContext)
                classifier = loaded
                runOnUiThread {
                    if (!isFinishing && !isDestroyed) {
                        btnPredict.isEnabled = true
                    }
                }
            } catch (e: Exception) {
                runOnUiThread {
                    if (!isFinishing && !isDestroyed) {
                        Toast.makeText(
                            this,
                            "No se pudo cargar modelo_linear_android.json: ${e.message}",
                            Toast.LENGTH_LONG
                        ).show()
                    }
                }
            }
        }
    }

    private fun runPrediction() {
        val activeClassifier = classifier
        if (activeClassifier == null) {
            Toast.makeText(this, R.string.msg_loading_model, Toast.LENGTH_SHORT).show()
            return
        }

        val rawBitmap = drawingView.getBitmap()
        val preprocessResult = MnistPreprocessor.process(rawBitmap)
        rawBitmap.recycle()

        when (preprocessResult) {
            is PreprocessResult.EmptyCanvas -> {
                Toast.makeText(this, R.string.msg_empty_canvas, Toast.LENGTH_SHORT).show()
            }

            is PreprocessResult.Success -> {
                val prediction = activeClassifier.predict(preprocessResult.features)
                displayPrediction(prediction, preprocessResult.previewBitmap)
            }
        }
    }

    private fun displayPrediction(
        prediction: SvmPredictionResult,
        preview28: Bitmap
    ) {
        tvPredictedDigit.text = prediction.predictedDigit.toString()
        tvConfidence.text = String.format(
            Locale.US,
            "Seguridad: %.1f%%",
            prediction.confidencePercent
        )
        tvLatency.text = String.format(
            Locale.US,
            "Tiempo: %.2f ms",
            prediction.inferenceTimeMs
        )

        val crispPreview = Bitmap.createScaledBitmap(preview28, 168, 168, false)
        if (preview28 !== crispPreview) {
            preview28.recycle()
        }

        val isNight = isNightModeActive()
        val displayBitmap = if (isNight) {
            crispPreview
        } else {
            val inverted = Bitmap.createBitmap(crispPreview.width, crispPreview.height, Bitmap.Config.ARGB_8888)
            val pixels = IntArray(crispPreview.width * crispPreview.height)
            crispPreview.getPixels(pixels, 0, crispPreview.width, 0, 0, crispPreview.width, crispPreview.height)
            for (i in pixels.indices) {
                val p = pixels[i]
                val r = 255 - ((p ushr 16) and 0xFF)
                val g = 255 - ((p ushr 8) and 0xFF)
                val b = 255 - (p and 0xFF)
                pixels[i] = (p and 0xFF000000.toInt()) or (r shl 16) or (g shl 8) or b
            }
            inverted.setPixels(pixels, 0, crispPreview.width, 0, 0, crispPreview.width, crispPreview.height)
            crispPreview.recycle()
            inverted
        }

        ivPreview28.setImageBitmap(displayBitmap)
        currentPreviewBitmap?.let { previous ->
            if (previous !== displayBitmap && !previous.isRecycled) {
                previous.recycle()
            }
        }
        currentPreviewBitmap = displayBitmap

        val top3 = prediction.probabilities.take(3)
        topRows.forEachIndexed { idx, row ->
            val item = top3.getOrNull(idx)
            if (item != null) {
                row.label.text = "${row.rank}°  ${item.digit}"
                row.progress.setProgressCompat((item.probabilityPercent * 10f).roundToInt(), true)
                row.percent.text = String.format(Locale.US, "%.1f%%", item.probabilityPercent)
            } else {
                row.label.text = "${row.rank}°  —"
                row.progress.setProgressCompat(0, false)
                row.percent.text = "0.0%"
            }
        }
    }

    private fun resetResults() {
        tvPredictedDigit.setText(R.string.placeholder_dash)
        tvConfidence.text = "Seguridad: —"
        tvLatency.text = "Tiempo: —"
        ivPreview28.setImageDrawable(null)
        currentPreviewBitmap?.let { previous ->
            if (!previous.isRecycled) {
                previous.recycle()
            }
        }
        currentPreviewBitmap = null

        topRows.forEach { row ->
            row.label.text = "${row.rank}° —"
            row.progress.setProgressCompat(0, false)
            row.percent.text = "0.0%"
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        currentPreviewBitmap?.let { bmp ->
            if (!bmp.isRecycled) bmp.recycle()
        }
        backgroundExecutor.shutdownNow()
    }
}
