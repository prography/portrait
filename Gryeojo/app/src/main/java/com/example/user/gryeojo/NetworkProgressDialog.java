package com.example.user.gryeojo;

import android.app.Activity;
import android.app.Dialog;
import android.content.Context;
import android.view.Window;

public class NetworkProgressDialog extends Dialog {
    private Context context;

    /**
     * 네트워크 프로그래스바 초기화
     * @param context
     */
    public NetworkProgressDialog(Context context) {
        super(context, android.R.style.Theme_Translucent_NoTitleBar);

        requestWindowFeature(Window.FEATURE_NO_TITLE);
        setContentView(R.layout.network_progress_dialog);

        this.context = context;

    }

    @Override
    public void show() {

        if(((Activity)context).isFinishing())
            return;

        super.show();
    }

    @Override
    public void dismiss() {
        super.dismiss();
    }
}
