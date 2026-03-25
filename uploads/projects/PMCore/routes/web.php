<?php

use Illuminate\Support\Facades\Route;
use Modules\PMCore\app\Http\Controllers\ProjectController;
use Modules\PMCore\app\Http\Controllers\ProjectDashboardController;
use Modules\PMCore\app\Http\Controllers\ProjectReportController;
use Modules\PMCore\app\Http\Controllers\ProjectStatusController;
use Modules\PMCore\app\Http\Controllers\ProjectTaskController;
use Modules\PMCore\app\Http\Controllers\ResourceController;
use Modules\PMCore\app\Http\Controllers\TimesheetController;

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

Route::group(['prefix' => 'projects', 'as' => 'pmcore.', 'middleware' => ['auth']], function () {
    // Dashboard
    Route::get('/dashboard', [ProjectDashboardController::class, 'index'])->name('dashboard.index');

    // Project Status Management (must be before /{project} route)
    Route::prefix('project-statuses')->as('project-statuses.')->group(function () {
        Route::get('/', [ProjectStatusController::class, 'index'])->name('index');
        Route::get('/data/ajax', [ProjectStatusController::class, 'getDataAjax'])->name('getDataAjax');
        Route::post('/', [ProjectStatusController::class, 'store'])->name('store');
        Route::get('/{projectStatus}', [ProjectStatusController::class, 'show'])->name('show');
        Route::put('/{projectStatus}', [ProjectStatusController::class, 'update'])->name('update');
        Route::delete('/{projectStatus}', [ProjectStatusController::class, 'destroy'])->name('destroy');
        Route::post('/{projectStatus}/toggle-active', [ProjectStatusController::class, 'toggleActive'])->name('toggle-active');
        Route::post('/sort-order', [ProjectStatusController::class, 'updateSortOrder'])->name('update-sort-order');
        Route::post('/set-default', [ProjectStatusController::class, 'setAsDefault'])->name('set-default');
    });

    // Timesheet Management (must be before /{project} route)
    Route::prefix('timesheets')->as('timesheets.')->group(function () {
        Route::get('/', [TimesheetController::class, 'index'])->name('index');
        Route::get('/data/ajax', [TimesheetController::class, 'indexAjax'])->name('data');
        Route::get('/create', [TimesheetController::class, 'create'])->name('create');
        Route::get('/statistics', [TimesheetController::class, 'statistics'])->name('statistics');
        Route::get('/projects/{project}/tasks', [TimesheetController::class, 'getProjectTasks'])->name('project-tasks');
        Route::post('/', [TimesheetController::class, 'store'])->name('store');
        Route::get('/{timesheet}', [TimesheetController::class, 'show'])->name('show');
        Route::get('/{timesheet}/edit', [TimesheetController::class, 'edit'])->name('edit');
        Route::put('/{timesheet}', [TimesheetController::class, 'update'])->name('update');
        Route::delete('/{timesheet}', [TimesheetController::class, 'destroy'])->name('destroy');
        Route::post('/{timesheet}/approve', [TimesheetController::class, 'approve'])->name('approve');
        Route::post('/{timesheet}/reject', [TimesheetController::class, 'reject'])->name('reject');
        Route::post('/{timesheet}/submit', [TimesheetController::class, 'submit'])->name('submit');
    });

    Route::get('/', [ProjectController::class, 'index'])->name('projects.index');
    Route::get('/create', [ProjectController::class, 'create'])->name('projects.create');
    Route::post('/', [ProjectController::class, 'store'])->name('projects.store');
    Route::get('/data/ajax', [ProjectController::class, 'getDataAjax'])->name('projects.data');

    // Project member management
    Route::post('/{project}/members', [ProjectController::class, 'addMember'])->name('projects.members.store');
    Route::get('/{project}/members/{member}', [ProjectController::class, 'getMemberDetails'])->name('projects.members.show');
    Route::delete('/{project}/members/{member}', [ProjectController::class, 'removeMember'])->name('projects.members.destroy');
    Route::put('/{project}/members/{member}', [ProjectController::class, 'updateMemberRole'])->name('projects.members.update');

    // User search for member assignment
    Route::get('/users/search', [ProjectController::class, 'searchUsers'])->name('users.search');

    // Client search for project assignment
    Route::get('/clients/search', [ProjectController::class, 'searchClients'])->name('clients.search');

    // Project search for timesheets and other features
    Route::get('/search', [ProjectController::class, 'searchProjects'])->name('projects.search');

    // Project Timesheets
    Route::get('/{project}/timesheets', [ProjectController::class, 'timesheets'])->name('projects.timesheets');

    // Resource Management (Phase 4)
    Route::prefix('resources')->as('resources.')->group(function () {
        Route::get('/', [ResourceController::class, 'index'])->name('index');
        Route::get('/data/ajax', [ResourceController::class, 'indexAjax'])->name('data');
        Route::get('/create', [ResourceController::class, 'create'])->name('create');
        Route::post('/', [ResourceController::class, 'store'])->name('store');
        Route::get('/capacity', [ResourceController::class, 'capacity'])->name('capacity');
        Route::get('/capacity/data', [ResourceController::class, 'capacityData'])->name('capacity.data');
        Route::get('/{user}/schedule', [ResourceController::class, 'schedule'])->name('schedule');
        Route::get('/{allocation}/edit', [ResourceController::class, 'edit'])->name('edit');
        Route::put('/{allocation}', [ResourceController::class, 'update'])->name('update');
        Route::delete('/{allocation}', [ResourceController::class, 'destroy'])->name('destroy');
        Route::post('/availability', [ResourceController::class, 'availability'])->name('availability');
    });

    // Project Tasks (Phase 2)
    Route::prefix('{project}/tasks')->as('projects.tasks.')->group(function () {
        Route::get('/', [ProjectTaskController::class, 'index'])->name('index');
        Route::get('/board', [ProjectTaskController::class, 'board'])->name('board');
        Route::get('/data/ajax', [ProjectTaskController::class, 'getDataAjax'])->name('getDataAjax');
        Route::post('/', [ProjectTaskController::class, 'store'])->name('store');
        Route::get('/{task}', [ProjectTaskController::class, 'show'])->name('show');
        Route::put('/{task}', [ProjectTaskController::class, 'update'])->name('update');
        Route::delete('/{task}', [ProjectTaskController::class, 'destroy'])->name('destroy');
        Route::post('/{task}/complete', [ProjectTaskController::class, 'complete'])->name('complete');
        Route::post('/{task}/start', [ProjectTaskController::class, 'start'])->name('start');
        Route::post('/{task}/stop', [ProjectTaskController::class, 'stop'])->name('stop');
        Route::post('/reorder', [ProjectTaskController::class, 'reorder'])->name('reorder');
    });

    // Reports (must be before /{project} routes)
    Route::prefix('reports')->as('reports.')->group(function () {
        Route::get('/', [ProjectReportController::class, 'index'])->name('index');
        Route::get('/time', [ProjectReportController::class, 'timeReport'])->name('time');
        Route::get('/budget', [ProjectReportController::class, 'budgetReport'])->name('budget');
        Route::get('/resource', [ProjectReportController::class, 'resourceReport'])->name('resource');
        Route::post('/export', [ProjectReportController::class, 'export'])->name('export');
    });

    // Project routes with parameters (must be after specific routes)
    Route::get('/{project}', [ProjectController::class, 'show'])->name('projects.show');
    Route::get('/{project}/edit', [ProjectController::class, 'edit'])->name('projects.edit');
    Route::put('/{project}', [ProjectController::class, 'update'])->name('projects.update');
    Route::delete('/{project}', [ProjectController::class, 'destroy'])->name('projects.destroy');
    Route::post('/{project}/duplicate', [ProjectController::class, 'duplicate'])->name('projects.duplicate');
    Route::post('/{project}/archive', [ProjectController::class, 'archive'])->name('projects.archive');
});
