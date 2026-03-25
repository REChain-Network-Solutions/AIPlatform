<?php

use Illuminate\Support\Facades\Route; 
use Modules\Engineerings\Http\Controllers\MeterController;
use Modules\Engineerings\Http\Controllers\RecurringWorkOrderController;
use Modules\Engineerings\Http\Controllers\ServicesCategoryController;
use Modules\Engineerings\Http\Controllers\ServicesController;
use Modules\Engineerings\Http\Controllers\ServicesSubCategoryController;
use Modules\Engineerings\Http\Controllers\WorkRequestController;
use Modules\Engineerings\Http\Controllers\WorkOrderController;
use Modules\Engineerings\Http\Controllers\WorkOrderFileController;
use Modules\Engineerings\Http\Controllers\WorkCalendarController;

/*
|--------------------------------------------------------------------------
| Web Routes
|--------------------------------------------------------------------------
|
| Here is where you can register web routes for your application. These
| routes are loaded by the RouteServiceProvider within a group which
| contains the "web" middleware group. Now create something great!
|
*/

Route::group(['middleware' => 'auth', 'prefix' => 'account'], function () {
    Route::get('engineerings/export', ['uses' => 'WorkRequestController@export'])->name('engineerings.export');
    Route::post('engineerings/apply-quick-action', [WorkRequestController::class, 'applyQuickAction'])->name('engineerings.apply_quick_action');
    Route::resource('engineerings', WorkRequestController::class);;
    Route::get('engineerings/download/{id}', [WorkRequestController::class, 'download'])->name('engineerings.download');
    Route::get('engineerings/get-service/{id}', [WorkRequestController::class, 'getService'])->name('engineerings.get_services');
    Route::get('engineerings/get-items/{id}', [WorkRequestController::class, 'getItem'])->name('engineerings.get_items');

    Route::get('work/export', ['uses' => 'WorkOrderController@export'])->name('work.export');
    Route::post('work/apply-quick-action', [WorkOrderController::class, 'applyQuickAction'])->name('work.apply_quick_action');
    Route::resource('work', WorkOrderController::class);;
    Route::get('work/download/{id}', [WorkOrderController::class, 'download'])->name('work.download');
    Route::post('work/multiple-upload', [WorkOrderFileController::class, 'storeMultiple'])->name('work.multiple_upload');
    Route::get('work/get-assets/{id}', [WorkOrderController::class, 'getAssets'])->name('work.get_assets');
    Route::resource('work-file', WorkOrderFileController::class);;

    Route::get('recurring-work/export', ['uses' => 'RecurringWorkOrderController@export'])->name('recurring-work.export');
    Route::post('recurring-work/apply-quick-action', [RecurringWorkOrderController::class, 'applyQuickAction'])->name('recurring-work.apply_quick_action');
    Route::post('recurring-work/change-status', [RecurringWorkOrderController::class, 'changeStatus'])->name('recurring-work.change_status');
    Route::resource('recurring-work', RecurringWorkOrderController::class);;

    Route::resource('work-calendar', WorkCalendarController::class);;

    Route::get('meter/export', ['uses' => 'MeterController@export'])->name('meter.export');
    Route::post('meter/apply-quick-action', [MeterController::class, 'applyQuickAction'])->name('meter.apply_quick_action');
    Route::post('meter/multiple-upload', [MeterController::class, 'storeMultiple'])->name('meter.multiple_upload');
    Route::post('meter/scan-barcode', [MeterController::class, 'scan'])->name('meter.scan_barcode');
    Route::resource('meter', MeterController::class);

    Route::resource('services', ServicesController::class);
    Route::resource('services-category', ServicesCategoryController::class);
    Route::get('get-services-subCategory/{id}', [ServicesSubCategoryController::class, 'getSubCategories'])->name('get_services_sub_category');
    Route::resource('sub-services-category', ServicesSubCategoryController::class);
});
