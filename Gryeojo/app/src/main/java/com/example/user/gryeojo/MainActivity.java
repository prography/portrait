package com.example.user.gryeojo;

import android.content.DialogInterface;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Build;
import android.preference.PreferenceManager;
import android.provider.MediaStore;
import android.support.v4.content.FileProvider;
import android.support.v7.app.AlertDialog;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Base64;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageButton;
import android.widget.ImageView;


import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;

public class MainActivity  extends AppCompatActivity {

    public static final int PICK_CAMERA = 101;

    String base_img;
    public static final int PICK_ALBUM = 102;
    Bitmap bitmap = null;
    private Uri mImageCaptureUri;
    private ImageButton inputImg;
    private boolean isSelectImage = false;
    SharedPreferences mPref;
    ImageView imageView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        inputImg = (ImageButton)findViewById(R.id.image_button);
        imageView = (ImageView)findViewById(R.id.imageView);
        inputImg.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                doTakeAlbumAction();
            }
        });

        mPref = PreferenceManager.getDefaultSharedPreferences(getApplicationContext());

    }



    //앨범에서 이미지 가져오는 것
    private void doTakeAlbumAction(){
        Intent intent = new Intent(Intent.ACTION_PICK);
        intent.setType(MediaStore.Images.Media.CONTENT_TYPE);
        intent.setData(MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
        startActivityForResult(intent, PICK_ALBUM);
    }

    public static String getBase64String(Bitmap bitmap)
    {
        ByteArrayOutputStream byteArrayOutputStream = new ByteArrayOutputStream();

        bitmap.compress(Bitmap.CompressFormat.JPEG, 100, byteArrayOutputStream);

        byte[] imageBytes = byteArrayOutputStream.toByteArray();

        return Base64.encodeToString(imageBytes, Base64.NO_WRAP);
    }

    private void bitmapImage(Uri uri){


        BitmapFactory.Options options = new BitmapFactory.Options();
        options.inPreferredConfig = Bitmap.Config.ARGB_8888;
        try {
            BitmapFactory.decodeStream(getContentResolver().openInputStream(uri), null, options);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
        try {
            bitmap = BitmapFactory.decodeStream(getContentResolver().openInputStream(uri), null, options);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        imageView.setImageBitmap(bitmap);
    }

    protected void onActivityResult(int requestCode, int resultCode, Intent data){
        switch (requestCode){
            case PICK_ALBUM:

                if(data == null){
                    break;
                }
                else{
                    Uri mImageCaptureUri = data.getData();
                    bitmapImage(mImageCaptureUri);
                    base_img = getBase64String(bitmap);
                    Log.d("base img",base_img);
                    SharedPreferences.Editor editor1 = mPref.edit();

                    editor1.putString("image", base_img);

                    editor1.commit();

                    Intent intent1 = new Intent(MainActivity.this, FirstActivity.class);
                    startActivity(intent1);

                }
                break;

        }
    }


    @Override
    public void onPointerCaptureChanged(boolean hasCapture) {

    }
}
