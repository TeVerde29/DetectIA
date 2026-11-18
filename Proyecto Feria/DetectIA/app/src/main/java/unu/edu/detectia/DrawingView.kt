package unu.edu.detectia

import android.content.Context
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
 * Lienzo táctil cuadrado con fondo negro (#000000) y trazo blanco (#FFFFFF)
 * grueso y suavizado, optimizado para capturar dígitos estilo MNIST.
 */
class DrawingView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private val strokePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
        strokeJoin = Paint.Join.ROUND
        isDither = true
        strokeWidth = 48f
    }

    private val currentPath = Path()
    private var bitmap: Bitmap? = null
    private var bitmapCanvas: Canvas? = null

    private var lastX = 0f
    private var lastY = 0f
    private var hasDrawn = false

    var onDrawStateChanged: ((hasContent: Boolean) -> Unit)? = null

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

        // Grosor proporcional al tamaño del lienzo (~8.5% del lado) para simular el trazo MNIST
        strokePaint.strokeWidth = max(18f, min(w, h) * 0.085f)

        val previousBitmap = bitmap
        val newBitmap = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val newCanvas = Canvas(newBitmap).apply {
            drawColor(Color.BLACK)
        }
        if (previousBitmap != null && hasDrawn) {
            val destRect = android.graphics.Rect(0, 0, w, h)
            newCanvas.drawBitmap(previousBitmap, null, destRect, null)
        }
        previousBitmap?.recycle()
        bitmap = newBitmap
        bitmapCanvas = newCanvas
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val currentBitmap = bitmap
        if (currentBitmap != null) {
            canvas.drawBitmap(currentBitmap, 0f, 0f, null)
        } else {
            canvas.drawColor(Color.BLACK)
        }
        canvas.drawPath(currentPath, strokePaint)
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        val x = event.x.coerceIn(0f, width.toFloat())
        val y = event.y.coerceIn(0f, height.toFloat())

        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                parent?.requestDisallowInterceptTouchEvent(true)
                currentPath.reset()
                currentPath.moveTo(x, y)
                // Pequeño desplazamiento para permitir trazos de un solo toque
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
                bitmapCanvas?.drawPath(currentPath, strokePaint)
                currentPath.reset()
                parent?.requestDisallowInterceptTouchEvent(false)
                invalidate()
                return true
            }
        }
        return super.onTouchEvent(event)
    }

    /**
     * Limpia completamente el lienzo volviéndolo a negro sólido.
     */
    fun clear() {
        currentPath.reset()
        bitmapCanvas?.drawColor(Color.BLACK)
        hasDrawn = false
        onDrawStateChanged?.invoke(false)
        invalidate()
    }

    /**
     * Devuelve una copia inmutable en ARGB_8888 del dibujo actual sobre fondo negro.
     */
    fun getBitmap(): Bitmap {
        val w = max(width, 1)
        val h = max(height, 1)
        val currentBitmap = bitmap
        if (currentBitmap != null) {
            val copy = currentBitmap.copy(Bitmap.Config.ARGB_8888, true)
            if (!currentPath.isEmpty) {
                Canvas(copy).drawPath(currentPath, strokePaint)
            }
            return copy
        }
        return Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888).apply {
            Canvas(this).drawColor(Color.BLACK)
        }
    }

    fun hasDrawing(): Boolean = hasDrawn
}
