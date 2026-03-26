<?php

use Illuminate\Support\Facades\Route;
use Modules\Inspection\Http\Controllers\ScheduleReplyController;
use Modules\Inspection\Http\Controllers\ScheduleController;
use Modules\Inspection\Http\Controllers\ScheduleFileController;
use Modules\Inspection\Http\Controllers\RecurringScheduleController;
use Modules\Inspection\Http\Controllers\ScheduleInspectionController;
use Modules\Inspection\Http\Controllers\InspectionTemplateController;
use Modules\Inspection\Http\Controllers\InspectionTemplateItemController;

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

    Route::group(['prefix' => 'schedules'], function () {
        // Schedule recurring

    });

    /*
     |--------------------------------------------------------------------------
     | Canonical, menu-safe resources
     |--------------------------------------------------------------------------
     |
     | Worksuite-style modules sometimes ship sidebar links that reference these
     | exact route names. We register them explicitly (and keep legacy URIs)
     | so the dashboard never hard-crashes with RouteNotFoundException.
     |
     */

    // Canonical (referenced by sidebar.blade.php)
    Route::resource('recurring-inspection_schedules', RecurringScheduleController::class)
        ->names('recurring-inspection_schedules');

    Route::resource('inspection_schedules', ScheduleController::class)
        ->names('inspection_schedules');

    // Legacy URIs (keep for backward compatibility / existing bookmarks)
    Route::resource('recurring-schedules', RecurringScheduleController::class);
    Route::post('recurring-schedule/change-status', [RecurringScheduleController::class, 'changeStatus'])->name('recurring_schedule.change_status');
    Route::get('recurring-schedule/export/{startDate}/{endDate}/{status}/{employee}', [RecurringScheduleController::class, 'export'])->name('recurring_schedule.export');
    Route::get('recurring-schedule/recurring-schedule/{id}', [RecurringScheduleController::class, 'recurringSchedules'])->name('recurring_schedule.recurring_schedule');

    Route::resource('schedules', ScheduleController::class);
    Route::post('schedules/apply-quick-action', [ScheduleController::class, 'applyQuickAction'])->name('schedules.apply_quick_action');
    Route::get('schedules/update-status/{scheduleID}', [ScheduleController::class, 'cancelStatus'])->name('schedules.update_status');

    Route::resource('schedule-inspection', ScheduleInspectionController::class);
    Route::post('schedule-inspection/refreshCount', [ScheduleInspectionController::class, 'refreshCount'])->name('schedule-inspection.refresh_count');
    Route::post('schedule-inspection/updateOtherData/{id}', [ScheduleInspectionController::class, 'updateOtherData'])->name('schedule-inspection.update_other_data');
    Route::post('inspection/apply-quick-action', [ScheduleInspectionController::class, 'applyQuickAction'])->name('inspection.apply_quick_action');
    Route::post('schedule-inspection/change-status', [ScheduleInspectionController::class, 'changeStatus'])->name('schedule-inspection.change-status');


    Route::get('schedule-files/download/{id}', [ScheduleFileController::class, 'download'])->name('schedule-files.download');
    
/*
 |--------------------------------------------------------------------------
 | Templates (Pass 4)
 |--------------------------------------------------------------------------
 | Keep this module strictly "inspections" by providing reusable checklists.
 | Compliance/SWMS live in Titan modules; templates simply standardise checks.
 */
Route::resource('inspection-templates', InspectionTemplateController::class)
    ->names('inspection-templates');

Route::post('inspection-templates/{template}/items', [InspectionTemplateItemController::class, 'store'])
    ->name('inspection-templates.items.store');
Route::delete('inspection-templates/{template}/items/{item}', [InspectionTemplateItemController::class, 'destroy'])
    ->name('inspection-templates.items.destroy');

Route::resource('schedule-files', ScheduleFileController::class);

    Route::resource('schedule-replies', ScheduleReplyController::class);
});
