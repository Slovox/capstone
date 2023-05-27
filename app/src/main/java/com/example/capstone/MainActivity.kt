package com.example.capstone

import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import com.example.capstone.databinding.ActivityMainBinding
import com.example.capstone.ml.Model
import com.google.android.material.button.MaterialButtonToggleGroup
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class MainActivity : AppCompatActivity() {

    private lateinit var binding : ActivityMainBinding
    //private lateinit var imageView: ImageView
    private lateinit var button: Button
    private lateinit var result: TextView
    private var GALLERY_REQUEST_CODE = 123

    lateinit var predBtn: Button
    lateinit var loadBtn: Button
    lateinit var resView: TextView
    lateinit var bitmap: Bitmap
    lateinit var imageView: ImageView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)
        val view = binding.root
        //setContentView(view)
        setContentView(R.layout.activity_main)

        imageView = binding.imageView
        button = binding.btnTakeImage
        result = binding.result

        val buttonLoad = binding.btnLoadImage
        loadBtn = findViewById(R.id.btn_load_image)
        predBtn = findViewById(R.id.predict)
        resView = findViewById(R.id.result)
        imageView = findViewById(R.id.imageView)



        var labels = application.assets.open("labels.txt").bufferedReader().readLines()

        //image processor
        var imageProcessor = ImageProcessor.Builder()
            .add(NormalizeOp(0.0f,255.0f))
            .add(ResizeOp(256,256,ResizeOp.ResizeMethod.BILINEAR))
            .build()

        //binding.btn_load_image.setOnClickListener { startGallery() }

        button.setOnClickListener{
            if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED)
            {
                takePicturePreview.launch(null)
            }
            else
            {
                requestPermission.launch(android.Manifest.permission.CAMERA)
            }
        }


        buttonLoad.setOnClickListener{
            if (ContextCompat.checkSelfPermission(this, android.Manifest.permission.READ_EXTERNAL_STORAGE)
                == PackageManager.PERMISSION_GRANTED){
                val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
                intent.type = "image/*"

                val mimeTypes = arrayOf("image/jpeg","image/jpg","image/png")
                intent.putExtra(Intent.EXTRA_MIME_TYPES, mimeTypes)
                intent.flags = Intent.FLAG_GRANT_READ_URI_PERMISSION
                onresult.launch(intent)
            }
            else {
                requestPermission.launch(android.Manifest.permission.READ_EXTERNAL_STORAGE)
            }
        }

        loadBtn.setOnClickListener{
            var intent = Intent()
            intent.setAction(Intent.ACTION_GET_CONTENT)
            intent.setType("image/*")
            startActivityForResult(intent,100)
        }
        //setContentView(R.layout.activity_main)

        predBtn.setOnClickListener{

            var tensorImage = TensorImage(DataType.FLOAT32)
            tensorImage.load(bitmap)

            val model = Model.newInstance(this)

            // Creates inputs for reference.
            val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 256, 256, 3), DataType.FLOAT32)
            inputFeature0.loadBuffer(tensorImage.buffer)

            // Runs model inference and gets result.
            val outputs = model.process(inputFeature0)
            val outputFeature0 = outputs.outputFeature0AsTensorBuffer.floatArray

            var maxIdx = 0
            outputFeature0.forEachIndexed{index, fl ->
                if (outputFeature0[maxIdx] < fl){
                    maxIdx = index
                }
            }
            result.setText(labels[maxIdx])
            // Releases model resources if no longer used.
            model.close()

        }
    }
    // request camera permission
    private val requestPermission = registerForActivityResult(ActivityResultContracts.RequestPermission()){granted->
        if (granted){
            takePicturePreview.launch(null)
        }else{
            Toast.makeText(this, "Permission Denied ! Try Again",Toast.LENGTH_SHORT).show()
        }
    }

    // launch camera and take picture
    private val takePicturePreview = registerForActivityResult(ActivityResultContracts.TakePicturePreview()){bitmap->
        if (bitmap != null){
            imageView.setImageBitmap(bitmap)
            outputGenerator(bitmap)
        }
    }

    // get image from gallery
    private val onresult = registerForActivityResult(ActivityResultContracts.StartActivityForResult()){result->
        Log.i("Tag","This is The Result: ${result.data} ${result.resultCode}")
        onResultReceived(GALLERY_REQUEST_CODE, result)
    }

    private fun onResultReceived(requestCode: Int, result: androidx.activity.result.ActivityResult){
        when(requestCode){
            GALLERY_REQUEST_CODE->{
                if(result?.resultCode == Activity.RESULT_OK){
                    result.data?.data?.let{uri->
                        Log.i("Tag","onResultReceived: $uri")

                        val bitmap = BitmapFactory.decodeStream(contentResolver.openInputStream(uri))
                        imageView.setImageBitmap(bitmap)
                        outputGenerator(bitmap)
                    }
                }
                else{
                    Log.e("Tag", "onActivityResult: Error in Selecting Image")
                }
            }
        }
    }

    private fun outputGenerator(bitmap: Bitmap){
        var tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(bitmap)

        var imageProcessor = ImageProcessor.Builder()
            .add(NormalizeOp(0.0f,255.0f))
            .add(ResizeOp(256,256,ResizeOp.ResizeMethod.BILINEAR))
            .build()
        tensorImage = imageProcessor.process(tensorImage)

        // model TFLite
        val model = Model.newInstance(this)

        // Creates inputs for reference.
        val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 256, 256, 3), DataType.FLOAT32)
        inputFeature0.loadBuffer(tensorImage.buffer)

        // Runs model inference and gets result.
        val outputs = model.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer.floatArray

        var maxIdx = 0
        outputFeature0.forEachIndexed{index, fl ->
            if (outputFeature0[maxIdx] < fl){
                maxIdx = index
            }
        }
        result.setText(maxIdx.toString())
        //result.text = labels[maxIdx]
        Log.i("Tag","outputGenerator: $maxIdx")
        // Releases model resources if no longer used.
        model.close()
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (requestCode == 100){
            var uri = data?.data
            bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)
            imageView.setImageBitmap(bitmap)
        }
    }


}