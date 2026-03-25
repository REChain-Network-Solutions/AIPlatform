<?php

use Illuminate\Support\Facades\Route;
use Modules\EInvoice\Http\Controllers\EInvoiceController;

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


// Admin routes
Route::group(['middleware' => 'auth', 'prefix' => 'account'], function () {
    Route::group(
        ['prefix' => 'settings'],
        function () {
            Route::get('einvoice', [EInvoiceController::class, 'settings'])->name('einvoice.settings');
            Route::get('einvoice-modal', [EInvoiceController::class, 'settingsModal'])->name('einvoice.settings_modal');
        }
    );
    Route::group(
        ['prefix' => 'einvoice', 'as' => 'einvoice.'],
        function () {
            Route::get('/', [EInvoiceController::class, 'index'])->name('index');
            Route::get('/export-xml/{id}', [EInvoiceController::class, 'exportXml'])->name('exportXml');
            Route::put('einvoice-save', [EInvoiceController::class, 'saveSettings'])->name('settings.save');
            Route::get('einvoice-client-modal/{id}', [EInvoiceController::class, 'clientModal'])->name('client_modal');
            Route::put('einvoice-client-save/{id}', [EInvoiceController::class, 'clientSave'])->name('client_save');
        }
    );
});

Route::middleware(['web','auth'])->group(function(){
    Route::get('/einvoice/ai/health', [\Modules\EInvoice\Http\Controllers\EInvoiceController::class, 'aiHealth'])->name('einvoice.ai.health');
});


Route::middleware(['web','auth'])->group(function(){
    Route::get('/einvoice/settings/ai', [\Modules\EInvoice\Http\Controllers\EInvoiceController::class, 'aiSettings'])->name('einvoice.settings.ai');
    Route::post('/einvoice/settings/ai/test', [\Modules\EInvoice\Http\Controllers\EInvoiceController::class, 'aiTest'])->name('einvoice.ai.test');
    Route::post('/einvoice/ai/generate/{invoice}', [\Modules\EInvoice\Http\Controllers\EInvoiceController::class, 'generateNote'])->name('einvoice.ai.generate');
});


Route::middleware(['web','auth'])->group(function(){
    Route::get('/einvoice/notes/latest/{invoice}', [\Modules\EInvoice\Http\Controllers\EInvoiceController::class, 'latestNote'])->name('einvoice.notes.latest');
});


Route::middleware(['web','auth'])->group(function(){
    Route::post('/einvoice/ai/draft', [\Modules\EInvoice\Http\Controllers\EInvoiceController::class, 'aiInvoiceDraft'])->name('einvoice.ai.draft');
    Route::post('/einvoice/ai/create-from-draft/{draft}', [\Modules\EInvoice\Http\Controllers\EInvoiceController::class, 'createInvoiceFromDraft'])->name('einvoice.ai.create_from_draft');
});

