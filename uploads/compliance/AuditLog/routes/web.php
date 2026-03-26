<?php

use Illuminate\Support\Facades\Route;
use Modules\AuditLog\app\Http\Controllers\AuditLogController;

/*
|--------------------------------------------------------------------------
| Web Routes
|--------------------------------------------------------------------------
*/

Route::prefix('auditlog')->name('auditlog.')->middleware(['auth', 'web'])->group(function () {
    Route::get('/', [AuditLogController::class, 'index'])->name('index');
    Route::get('/datatable', [AuditLogController::class, 'indexAjax'])->name('datatable');
    Route::get('/statistics', [AuditLogController::class, 'statistics'])->name('statistics');
    Route::get('/filters', [AuditLogController::class, 'filters'])->name('filters');
    Route::get('/{id}', [AuditLogController::class, 'show'])->name('show');
});
