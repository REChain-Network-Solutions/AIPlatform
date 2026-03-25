<?php
use Illuminate\Support\Facades\Route; use Modules\Expenses\Http\Controllers\ExpensesController;
Route::middleware(['web','auth','can:expenses.access'])->prefix('expenses')->name('expenses.')->group(function(){ Route::get('/',[ExpensesController::class,'index'])->name('index'); Route::get('/create',[ExpensesController::class,'create'])->name('create'); Route::post('/',[ExpensesController::class,'store'])->name('store');});

use Modules\Expenses\Http\Controllers\ApprovalsController;
use Modules\Expenses\Http\Controllers\ReceiptsController;
use Modules\Expenses\Http\Controllers\ExpensePagesController;

Route::middleware(['web','auth','can:expenses.access'])->prefix('expenses')->group(function(){
  Route::get('/{id}', [ExpensePagesController::class, 'show'])->name('show');
  Route::post('/{id}/submit', [ApprovalsController::class, 'submit'])->name('submit');
  Route::post('/{id}/approve', [ApprovalsController::class, 'approve'])->name('approve');
  Route::post('/{id}/reimburse', [ApprovalsController::class, 'reimburse'])->name('reimburse');
  Route::post('/{id}/receipts', [ReceiptsController::class, 'upload'])->name('receipts.upload');
});
