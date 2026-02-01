<?php

use App\Extensions\Announcement\System\Http\Controllers\AnnouncementController;
use Illuminate\Support\Facades\Route;

Route::middleware(['web', 'auth'])->prefix('dashboard/admin/public-announcement')->name('dashboard.admin.public-announcement.')->group(function () {
    Route::get('/create', [AnnouncementController::class, 'create'])->name('create');
    Route::post('/store', [AnnouncementController::class, 'store'])->name('store');
    Route::get('/edit/{announcement}', [AnnouncementController::class, 'edit'])->name('edit');
    Route::put('/update/{announcement}', [AnnouncementController::class, 'update'])->name('update');
    Route::get('/delete/{announcement}', [AnnouncementController::class, 'destroy'])->name('delete');
});
