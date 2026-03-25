<?php

use Illuminate\Support\Facades\Route;

Route::middleware(['api'])->group(function(){
    // Optional: add JSON endpoints for quotes if needed later.
});


use Modules\Quotes\Http\Controllers\QuoteController;

Route::get('/quotes/items/search', [QuoteController::class, 'searchItems'])->name('quotes.api.items.search');


use Modules\Quotes\Entities\PriceList;
Route::get('/quotes/pricelists', function(\Illuminate\Http\Request $request){
    $currency = (string) $request->query('currency', '');
    $q = PriceList::query();
    if ($currency !== '') $q->where('currency', $currency);
    return $q->orderBy('name')->get(['id','name','currency']);
})->name('quotes.api.pricelists');
