package com.example.user.gryeojo;

import android.app.Activity;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.preference.PreferenceManager;
import android.support.v7.app.AppCompatActivity;
import android.support.v7.widget.Toolbar;
import android.util.Base64;
import android.util.Log;
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

import java.io.ByteArrayOutputStream;

import static com.example.user.gryeojo.MainActivity.getBase64String;

public class FirstActivity extends AppCompatActivity {
    ImageView inputImageView;
    ImageButton imageButton1;
    ImageButton imageButton2;
    ImageButton imageButton3;
    ImageButton imageButton4;
    Bitmap decodedBitmap;
    String base_str;
    String input;
    String filter;
    String base_img;
    SharedPreferences mPref;
    Bitmap inputImage;
    Bitmap outputImage1, outputImage2, outputImage3, outputImage4;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_first);
        inputImageView = findViewById(R.id.input_imageView);
        Toolbar myToolbar = findViewById(R.id.my_toolbar);
        setSupportActionBar(myToolbar);

        mPref = PreferenceManager.getDefaultSharedPreferences(getApplicationContext());
        base_img = mPref.getString("image", "0");


        byte[] decodedByteArray = Base64.decode(base_img, Base64.NO_WRAP);
        decodedBitmap = BitmapFactory.decodeByteArray(decodedByteArray, 0, decodedByteArray.length);
        decodedBitmap =decodedBitmap.copy(Bitmap.Config.ARGB_8888, true);

        Log.d("image", String.valueOf(decodedBitmap));

        inputImageView.setImageBitmap(decodedBitmap);

        imageButton1 = findViewById(R.id.origin_btn);
        imageButton2 = findViewById(R.id.first_btn);
        imageButton3 = findViewById(R.id.second_btn);
        imageButton4 = findViewById(R.id.third_btn);

        BitmapFactory.Options opts = new BitmapFactory.Options();
        opts.inMutable = true;



        imageButton1.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                inputImageView.setImageBitmap(outputImage1);
            }
        });

        imageButton2.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
               filter = "water";
                result(filter);
            }
        });
        imageButton3.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
               filter = "oil";
                result(filter);
            }
        });
        imageButton4.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                filter = "pop";
                result(filter);
            }
        });
    }


    public void result(String filter){

        base_str = getBase64String(decodedBitmap);

        SharedPreferences.Editor editor = mPref.edit();

        editor.putString("img", base_str);

        editor.putString("key",String.valueOf(System.currentTimeMillis()));

        editor.putString("filter", filter);
        editor.commit();

        Intent intent = new Intent(FirstActivity.this, ResultActivity.class);
        startActivity(intent);


        Intent intent2 = new Intent(FirstActivity.this, ResultActivity.class);
        startActivity(intent2);
   }


    // method for base64 to bitmap
    public static Bitmap decodeBase64(String input) {
        byte[] decodedByte = Base64.decode(input, 0);
        return BitmapFactory
                .decodeByteArray(decodedByte, 0, decodedByte.length);
    }

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
