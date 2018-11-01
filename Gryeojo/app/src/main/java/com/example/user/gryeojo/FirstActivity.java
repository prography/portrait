package com.example.user.gryeojo;

import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.util.Base64;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.View;
import android.widget.ImageButton;
import android.widget.ImageView;
import android.widget.Toast;

import com.zomato.photofilters.geometry.Point;
import com.zomato.photofilters.imageprocessors.Filter;
import com.zomato.photofilters.imageprocessors.subfilters.BrightnessSubfilter;
import com.zomato.photofilters.imageprocessors.subfilters.ContrastSubfilter;
import com.zomato.photofilters.imageprocessors.subfilters.SaturationSubfilter;
import com.zomato.photofilters.imageprocessors.subfilters.ToneCurveSubfilter;

public class FirstActivity extends AppCompatActivity {
    ImageView inputImageView;
    ImageButton imageButton1;
    ImageButton imageButton2;
    ImageButton imageButton3;
    ImageButton imageButton4;


    Bitmap inputImage;
    Bitmap outputImage1, outputImage2, outputImage3, outputImage4;


    static
    {
        System.loadLibrary("NativeImageProcessor");
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_first);
        Toolbar myToolbar = findViewById(R.id.my_toolbar);
        setSupportActionBar(myToolbar);
        SharedPreferences result = getSharedPreferences("saveData", MODE_PRIVATE);
        String input = result.getString("imagePreference", "NOT FOUND");
        // Add back button
//        getSupportActionBar().setDisplayShowCustomEnabled(true);
//        getSupportActionBar().setDisplayHomeAsUpEnabled(true);

        inputImage = decodeBase64(input);
        inputImageView = findViewById(R.id.input_imageView);
        imageButton1 = findViewById(R.id.origin_btn);
        imageButton2 = findViewById(R.id.first_btn);
        imageButton3 = findViewById(R.id.second_btn);
        imageButton4 = findViewById(R.id.third_btn);

        BitmapFactory.Options opts = new BitmapFactory.Options();
        opts.inMutable = true;

        inputImageView.setImageBitmap(inputImage);

        outputImage1 = inputImage.copy(inputImage.getConfig(), true);
        outputImage2 = inputImage.copy(inputImage.getConfig(), true);
        outputImage3 = inputImage.copy(inputImage.getConfig(), true);
        outputImage4 = inputImage.copy(inputImage.getConfig(), true);

        outputImage2 = ToneCurveSubfilter(outputImage2);
        outputImage3 = SaturationSubfilter(outputImage3);
        outputImage4 = ContrastSubfilter(outputImage4);

        imageButton1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                inputImageView.setImageBitmap(outputImage1);
            }
        });

        imageButton2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                inputImageView.setImageBitmap(outputImage2);
            }
        });
        imageButton3.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                inputImageView.setImageBitmap(outputImage3);
            }
        });
        imageButton4.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                inputImageView.setImageBitmap(outputImage4);
            }
        });
    }


    public Bitmap Myfilter(Bitmap input) {
        Filter myFilter = new Filter();
        myFilter.addSubFilter(new BrightnessSubfilter(20));
        myFilter.addSubFilter(new ContrastSubfilter(0.5f));
        Bitmap output = myFilter.processFilter(input);
        return output;
    }
    public Bitmap ToneCurveSubfilter(Bitmap input) {
        Filter Filter1 = new Filter();
        Point[] rgbKnots;
        rgbKnots = new Point[3];
        rgbKnots[0] = new Point(0, 0);
        rgbKnots[1] = new Point(175, 139);
        rgbKnots[2] = new Point(255, 255);

        Filter1.addSubFilter(new ToneCurveSubfilter(rgbKnots, null, null, null));
        Bitmap output = Filter1.processFilter(input);

        return output;
    }

    public Bitmap SaturationSubfilter(Bitmap input) {

        Filter Filter2 = new Filter();
        Filter2.addSubFilter(new SaturationSubfilter(1.3f));
        Bitmap output = Filter2.processFilter(input);

        return output;
    }

    public Bitmap ContrastSubfilter(Bitmap input) {

        Filter Filter3 = new Filter();
        Filter3.addSubFilter(new ContrastSubfilter(1.2f));
        Bitmap output = Filter3.processFilter(input);

        return output;
    }

    // method for base64 to bitmap
    public static Bitmap decodeBase64(String input) {
        byte[] decodedByte = Base64.decode(input, 0);
        return BitmapFactory
                .decodeByteArray(decodedByte, 0, decodedByte.length);
    }


//    @Override
//    public boolean onOptionsItemSelected(MenuItem item){
//        int id = item.getItemId();
//        if(id == android.R.id.home) {
//            this.finish();
//        }
//        return super.onOptionsItemSelected(item);
//    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        //return super.onCreateOptionsMenu(menu);
        MenuInflater menuInflater = getMenuInflater();
        menuInflater.inflate(R.menu.menu, menu);
        return true;
    }
    //추가된 소스, ToolBar에 추가된 항목의 select 이벤트를 처리하는 함수
    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        finish();
        //return super.onOptionsItemSelected(item);
        switch (item.getItemId()) {
            case R.id.save_photo:
                // User chose the "Settings" item, show the app settings UI...
                Toast.makeText(getApplicationContext(), "저장되었습니다.", Toast.LENGTH_SHORT).show();
                return true;

            default:
                // If we got here, the user's action was not recognized.
                // Invoke the superclass to handle it.
                Toast.makeText(getApplicationContext(), "디폴트!!!", Toast.LENGTH_LONG).show();
                return super.onOptionsItemSelected(item);

        }
    }
}

/* https://stackoverflow.com/questions/18072448/how-to-save-image-in-shared-preference-in-android-shared-preference-issue-in-a */
