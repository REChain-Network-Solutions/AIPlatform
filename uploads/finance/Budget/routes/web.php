<?php

use Illuminate\Support\Facades\Route;

Route::middleware(['web', 'auth'])->group(function() {
    Route::get('/budgetallocationaprovalmodule', function() {
        return view('budgetallocationaprovalmodule::index');
    })->name('budgetallocationaprovalmodule.index');
});
