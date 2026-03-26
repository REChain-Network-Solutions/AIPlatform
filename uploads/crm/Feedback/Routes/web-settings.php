<?php

use Illuminate\Support\Facades\Route;
use Modules\Feedback\Http\Controllers\FeedbackSettingController;

Route::group(['middleware' => 'auth', 'prefix' => 'account/settings'], function () {
    Route::resource('feedback-settings', FeedbackSettingController::class);
});
