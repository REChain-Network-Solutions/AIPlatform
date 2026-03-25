<?php
use Illuminate\Support\Facades\Route; Route::middleware(['web','auth','can:expenses.access'])->prefix('expenses/ai')->group(function(){ Route::post('/categorize', fn()=>response()->json(['ok':true,'text'=>'AI categorize placeholder'])); });
Route::post('/receipt-ocr', fn() => response()->json(['ok'=>true,'text'=>'AI OCR placeholder']));
