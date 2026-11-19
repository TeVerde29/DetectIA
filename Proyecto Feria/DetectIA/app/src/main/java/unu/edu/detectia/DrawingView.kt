package unu.edu.detectia

import android.content.Context
import android.content.res.Configuration
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View
import kotlin.math.max
import kotlin.math.min

/**
 * Lienzo táctil cuadrado adaptable con tema:
 * - Modo Claro: Fondo blanco (#FFFFFF) y trazo negro (#1F1F1F)
 * - Modo Oscuro: Fondo oscuro (#1E1F20) y trazo blanco (#FFFFFF)
 * Internamente genera una copia estandarizada con fondo negro y trazo blanco
 * para garantizar la compatibilidad matemática con el modelo MNIST (28x28).
 */
class DrawingView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private var isDarkMode: Boolean = false

    private val strokePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
        isDither = true
        strokeWidth = 48f
    }

    private val recordedPath = Path()
    private val currentPath = Path()

    private var lastX = 0f
    private var lastY = 0f
    private var hasDrawn = false

    var onDrawStateChanged: ((hasContent: Boolean) -> Unit)? = null

    init {
        val uiMode = context.resources.configuration.uiMode and Configuration.UI_MODE_NIGHT_MASK
        isDarkMode = (uiMode == Configuration.UI_MODE_NIGHT_YES)
        updatePaintColor()
    }

    fun setNightMode(isNight: Boolean) {
        if (isDarkMode != isNight) {
            isDarkMode = isNight
            updatePaintColor()
            invalidate()
        }
    }

    private fun updatePaintColor() {
        strokePaint.color = if (isDarkMode) Color.WHITE else Color.parseColor("#1F1F1F")
    }

    override fun onMeasure(widthMeasureSpec: Int, heightMeasureSpec: Int) {
        super.onMeasure(widthMeasureSpec, heightMeasureSpec)
        val width = measuredWidth
        val height = measuredHeight
        val size = when {
            width > 0 && height > 0 -> min(width, height)
            width > 0 -> width
            else -> height
        }.coerceAtLeast(1)
        setMeasuredDimension(size, size)
    }

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) {
        super.onSizeChanged(w, h, oldw, oldh)
        if (w <= 0 || h <= 0) return
        // Grosor proporcional al tamaño del lienzo (~8.5% del lado) para emular trazo MNIST
        strokePaint.strokeWidth = max(18f, min(w, h) * 0.085f)
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val bgColor = if (isDarkMode) Color.parseColor("#1E1F20") else Color.WHITE
        canvas.drawColor(bgColor)

        updatePaintColor()
        canvas.drawPath(recordedPath, strokePaint)
        if (!currentPath.isEmpty) {
            canvas.drawPath(currentPath, strokePaint)
        }
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        val x = event.x.coerceIn(0f, width.toFloat())
        val y = event.y.coerceIn(0f, height.toFloat())

        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                parent?.requestDisallowInterceptTouchEvent(true)
                currentPath.reset()
                currentPath.moveTo(x, y)
                currentPath.lineTo(x + 0.1f, y + 0.1f)
                lastX = x
                lastY = y
                if (!hasDrawn) {
                    hasDrawn = true
                    onDrawStateChanged?.invoke(true)
                }
                invalidate()
                return true
            }

            MotionEvent.ACTION_MOVE -> {
                parent?.requestDisallowInterceptTouchEvent(true)
                val midX = (lastX + x) / 2f
                val midY = (lastY + y) / 2f
                currentPath.quadTo(lastX, lastY, midX, midY)
                lastX = x
                lastY = y
                invalidate()
                return true
            }

            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                currentPath.lineTo(x, y)
                recordedPath.addPath(currentPath)
                currentPath.reset()
                parent?.requestDisallowInterceptTouchEvent(false)
                invalidate()
                return true
            }
        }
        return super.onTouchEvent(event)
    }

    /**
     * Limpia completamente el lienzo manteniendo el color de fondo correspondiente.
     */
    fun clear() {
        recordedPath.reset()
        currentPath.reset()
        hasDrawn = false
        onDrawStateChanged?.invoke(false)
        invalidate()
    }

    /**
     * Devuelve una copia en ARGB_8888 con fondo negro (#000000) y trazo blanco (#FFFFFF)
     * para la inferencia de MNIST por el clasificador SVM.
     */
    fun getBitmap(): Bitmap {
        val w = max(width, 1)
        val h = max(height, 1)
        val mnistBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val mnistCanvas = Canvas(mnistBitmap).apply {
            drawColor(Color.BLACK)
        }

        val mnistPaint = Paint(strokePaint).apply {
            color = Color.WHITE
        }

        mnistCanvas.drawPath(recordedPath, mnistPaint)
        if (!currentPath.isEmpty) {
            mnistCanvas.drawPath(currentPath, mnistPaint)
        }

        return mnistBitmap
    }

    fun hasDrawing(): Boolean = hasDrawn
}
