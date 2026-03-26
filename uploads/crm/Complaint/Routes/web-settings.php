<?php

use Illuminate\Support\Facades\Route;
use Modules\Complaint\Http\Controllers\ComplaintSettingController;

Route::group(['middleware' => 'auth', 'prefix' => 'account/settings'], function () {
    Route::resource('complaint-settings', ComplaintSettingController::class);
});
