<?php
use Illuminate\Support\Facades\Route; use Modules\Treasury\Http\Controllers\TreasuryController;
Route::middleware(['web','auth','can:treasury.access'])->prefix('treasury')->name('treasury.')->group(function(){Route::get('/',[TreasuryController::class,'index'])->name('index');Route::post('/accounts',[TreasuryController::class,'createAccount'])->name('accounts.create');});

use Modules\Treasury\Http\Controllers\BankFeedController;
use Modules\Treasury\Http\Controllers\PaymentRunsController;

Route::middleware(['web','auth','can:treasury.access'])->prefix('treasury')->group(function () {
    Route::post('/feed/upload', [BankFeedController::class, 'upload'])->name('feed.upload');
    Route::post('/payment-runs', [PaymentRunsController::class, 'create'])->name('paymentruns.create');
    Route::get('/payment-runs/{id}', [PaymentRunsController::class, 'show'])->name('paymentruns.show');
});

use Modules\Treasury\Http\Controllers\ExportsController;
Route::get('/treasury/payment-runs/{id}/export/aba', [ExportsController::class, 'aba'])
  ->middleware(['web','auth','can:treasury.access'])->name('treasury.paymentruns.export.aba');

use Modules\Treasury\Http\Controllers\RulesController;
Route::middleware(['web','auth','can:treasury.access'])->group(function () {
    Route::get('/treasury/rules', [RulesController::class, 'index'])->name('treasury.rules.index');
    Route::post('/treasury/rules', [RulesController::class, 'store'])->name('treasury.rules.store');
});

use Modules\Treasury\Http\Controllers\ExportsController as X;
Route::get('/treasury/payment-runs/{id}/export/sepa', [X::class, 'sepa'])->middleware(['web','auth','can:treasury.access'])->name('treasury.paymentruns.export.sepa');
Route::get('/treasury/payment-runs/{id}/export/csv', [X::class, 'csv'])->middleware(['web','auth','can:treasury.access'])->name('treasury.paymentruns.export.csv');

use Modules\Treasury\Http\Controllers\PaymentRunPagesController as PRP;
Route::get('/treasury/payment-runs', [PRP::class, 'index'])->middleware(['web','auth','can:treasury.access'])->name('treasury.runs.index');
Route::get('/treasury/payment-runs/{id}/view', [PRP::class, 'show'])->middleware(['web','auth','can:treasury.access'])->name('treasury.runs.view');
