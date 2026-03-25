<?php

use Illuminate\Support\Facades\Route;
use Modules\BudgetAllocationAprovalModule\Http\Controllers\BudgetAllocationModuleController;

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

Route::group([['middleware' => 'auth', 'prefix' => 'budgetallocation']], function () {
    Route::resource('budgetallocationaprovalmodule', BudgetAllocationModuleController::class)->names('budgetallocationaprovalmodule');
});
