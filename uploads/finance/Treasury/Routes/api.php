<?php
use Illuminate\Support\Facades\Route; use Modules\Treasury\Http\Controllers\Api\TreasuryApiController;
Route::middleware(['api','auth:sanctum','can:treasury.access'])->prefix('api/treasury')->name('treasury.api.')->group(function(){Route::get('/accounts',[TreasuryApiController::class,'accounts'])->name('accounts');Route::get('/unreconciled',[TreasuryApiController::class,'unreconciled'])->name('unreconciled');});

use Modules\Treasury\Http\Controllers\ReconciliationController;
Route::post('/api/treasury/match-suggest', [ReconciliationController::class, 'suggest'])->middleware(['auth:sanctum','can:treasury.access'])->name('treasury.api.match');
