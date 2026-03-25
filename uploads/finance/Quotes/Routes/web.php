<?php

use Illuminate\Support\Facades\Route;
use Modules\Quotes\Http\Controllers\QuoteController;

Route::middleware(['web','auth'])->group(function(){
    Route::get('/quotes', [QuoteController::class, 'index'])->name('quotes.index');
    Route::get('/quotes/create', [QuoteController::class, 'create'])->name('quotes.create');
    Route::post('/quotes', [QuoteController::class, 'store'])->name('quotes.store');
    Route::get('/quotes/{id}', [QuoteController::class, 'show'])->name('quotes.show');
    Route::get('/quotes/{id}/edit', [QuoteController::class, 'edit'])->name('quotes.edit');
    Route::put('/quotes/{id}', [QuoteController::class, 'update'])->name('quotes.update');
    Route::delete('/quotes/{id}', [QuoteController::class, 'destroy'])->name('quotes.destroy');
});

Route::middleware(['web','auth'])->group(function(){
    Route::get('/quotes/{id}/pdf', [QuoteController::class, 'pdf'])->name('quotes.pdf');
    Route::post('/quotes/{id}/send', [QuoteController::class, 'send'])->name('quotes.send');
    Route::post('/quotes/{id}/convert', [QuoteController::class, 'convert'])->name('quotes.convert');
});


use Illuminate\Support\Facades\URL;
use Modules\Quotes\Http\Controllers\QuoteController as PublicQuoteController;

Route::middleware(['web','signed'])->group(function(){
    Route::get('/quotes/public/{id}', [PublicQuoteController::class, 'publicShow'])->name('quotes.public.show');
    Route::post('/quotes/public/{id}/{action}', [PublicQuoteController::class, 'publicAction'])->name('quotes.public.action');
});
