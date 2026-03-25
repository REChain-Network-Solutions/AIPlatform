<?php

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

use Illuminate\Support\Facades\Route;
use Workdo\Pos\Http\Controllers\PosController;
use Workdo\Pos\Http\Controllers\ReportController;

Route::group(['middleware' => ['web', 'auth', 'verified','PlanModuleCheck:Pos']], function ()
{

        Route::get('dashboard/pos',[PosController::class, 'dashboard'])->name('pos.dashboard');
        Route::post('pos/setting/store', [PosController::class, 'setting'])->name('pos.setting.store');
        Route::resource('pos', PosController::class);
        Route::get('pos-grid', [PosController::class, 'grid'])->name('pos.grid');
        Route::get('report/pos', [PosController::class, 'report'])->name('pos.report');
        Route::get('search-products', [PosController::class, 'searchProducts'])->name('search.products');
        Route::get('name-search-products', [PosController::class, 'searchProductsByName'])->name('name.search.products');
        Route::post('warehouse-empty-cart', [PosController::class, 'warehouseemptyCart'])->name('warehouse-empty-cart');
        Route::get('product-categories', [PosController::class, 'getProductCategories'])->name('product.categories');
        Route::post('empty-cart', [PosController::class, 'emptyCart']);
        Route::get('add-to-cart/{id}/{session}/{war_id}', [PosController::class, 'addToCart']);
        Route::delete('remove-from-cart', [PosController::class, 'removeFromCart']);
        Route::patch('update-cart', [PosController::class, 'updateCart']);

        Route::get('pos/data/store', [PosController::class, 'store'])->name('pos.data.store');

        // thermal print
        Route::get('printview/pos', [PosController::class, 'printView'])->name('pos.printview');

        Route::post('/cartdiscount', [PosController::class, 'cartdiscount'])->name('cartdiscount');

        Route::get('pos/pdf/{id}', [PosController::class, 'pos'])->name('pos.pdf');
        Route::post('/pos/template/setting', [PosController::class, 'savePosTemplateSettings'])->name('pos.template.setting');
        Route::get('pos/preview/{template}/{color}', [PosController::class, 'previewPos'])->name('pos.preview');


        //Reports
        Route::get('reports-daily-pos', [ReportController::class, 'posDailyReport'])->name('report.daily.pos');
        Route::get('reports-monthly-pos', [ReportController::class, 'posMonthlyReport'])->name('report.monthly.pos');
        Route::get('reports-pos-vs-purchase', [ReportController::class, 'posVsPurchaseReport'])->name('report.pos.vs.purchase');

        //pos barcode
        Route::get('barcode/pos', [PosController::class, 'barcode'])->name('pos.barcode');
        Route::get('setting/pos', [PosController::class, 'barcodeSetting'])->name('pos.setting');
        Route::post('barcode/settings', [PosController::class, 'BarcodesettingStore'])->name('barcode.setting');
        Route::get('print/pos', [PosController::class, 'printBarcode'])->name('pos.print');
        Route::post('pos/getproduct', [PosController::class, 'getproduct'])->name('pos.getproduct');
        Route::any('pos-receipt', [PosController::class, 'receipt'])->name('pos.receipt');

});
