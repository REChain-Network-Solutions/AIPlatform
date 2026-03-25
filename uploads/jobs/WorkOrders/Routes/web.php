<?php

use Illuminate\Support\Facades\Route;
use Modules\WorkOrders\Http\Controllers\FsmSettingsController;

// Import controllers once, with unique aliases to avoid clashes with other modules
use Modules\WorkOrders\Http\Controllers\{
    DashboardController,
    ChecklistController,
    ClientPortalController as WorkOrdersClientPortalController,
    ClientSignController   as WorkOrdersClientSignController,
    QaPdfController,
    CommentsController,
    RecurrenceController,
    IcsExportController,
    WorkOrderStatusController
};

/**
 * Admin / staff (authenticated users)
 */
Route::middleware(['web','auth'])->group(function () {
    // FSM settings
    Route::get('/admin/fsm-settings',  [FsmSettingsController::class,'index'])->name('workorders.admin.fsm');
    Route::post('/admin/fsm-settings', [FsmSettingsController::class,'save'])->name('workorders.admin.fsm.save');

    // Internal dashboard + workorder actions
    Route::get('/fsm/dashboard', [DashboardController::class,'index'])->name('workorders.dashboard');

    Route::get('/workorders/{id}/checklists/picker', [ChecklistController::class,'picker'])->name('workorders.checklists.picker');
    Route::post('/workorders/{id}/checklists/attach', [ChecklistController::class,'attach'])->name('workorders.checklists.attach');
    Route::get('/workorders/{id}/checklists', [ChecklistController::class,'view'])->name('workorders.checklists.view');
    Route::post('/workorders/{id}/checklists/items/{itemId}', [ChecklistController::class,'updateItem'])->name('workorders.checklists.items.update');

    Route::get('/workorders/{id}/qa.pdf', [QaPdfController::class,'show'])->name('workorders.qa.pdf');

    Route::get('/workorders/{id}/comments',  [CommentsController::class,'index'])->name('workorders.widgets.comments');
    Route::post('/workorders/{id}/comments', [CommentsController::class,'store'])->name('workorders.comments.store');

    Route::post('/workorders/{id}/recurrence', [RecurrenceController::class,'store'])->name('workorders.recurrence.store');

    Route::get('/workorders/{id}/ics',       [IcsExportController::class,'workOrder'])->name('workorders.ics');
    Route::get('/contractors/schedule/ics',  [IcsExportController::class,'contractor'])->name('contractors.schedule.ics');

    Route::post('/workorders/{id}/complete', [WorkOrderStatusController::class,'complete'])->name('workorders.complete');
});

/**
 * Client portal (customer-facing). Use a distinct alias to avoid cross-module collisions.
 */
Route::middleware(['web','auth-client'])->prefix('account')->group(function () {
    Route::get('/workorders',            [WorkOrdersClientPortalController::class,'index'])->name('client.workorders.index');
    Route::get('/workorders/{id}',       [WorkOrdersClientPortalController::class,'show'])->name('client.workorders.show');
    Route::get('/workorders/{id}/sign',  [WorkOrdersClientSignController::class,'form'])->name('client.workorders.sign.form');
    Route::post('/workorders/{id}/sign', [WorkOrdersClientSignController::class,'submit'])->name('client.workorders.sign');
});

// Optional legacy aliases (only keep if you truly need them, otherwise remove to avoid duplicate route names)
Route::middleware(['web','auth-client'])->group(function () {
    Route::get('/client/workorders',                 [WorkOrdersClientPortalController::class,'index']);
    Route::get('/client/workorders/{id}',            [WorkOrdersClientPortalController::class,'show']);
    Route::get('/client/workorders/{id}/sign',       [WorkOrdersClientSignController::class,'form']);
    Route::post('/client/workorders/{id}/sign',      [WorkOrdersClientSignController::class,'submit']);
});