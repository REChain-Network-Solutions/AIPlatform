<?php
namespace Modules\Expenses\Http\Controllers;
use Illuminate\Routing\Controller;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Storage;
use Modules\Expenses\Models\Expense;
use Modules\Expenses\Models\Receipt;
use Modules\Expenses\Services\AiClient;

class ReceiptsController extends Controller
{
  public function upload(Request $r, $id) {
    $e = Expense::findOrFail($id);
    $r->validate(['file' => 'required|file|max:10240']); // 10MB
    $f = $r->file('file');
    $path = $f->store('receipts','public');
    $rec = Receipt::create([
      'expense_id' => $e->id,
      'path' => $path,
      'mime' => $f->getClientMimeType(),
      'size' => $f->getSize(),
    ]);
    // optional AI OCR stub
    if (class_exists(AiClient::class)) {
      try {
        $text = app(AiClient::class)->__call('chat', ['expenses.receipt.ocr', ['expense_id'=>$e->id, 'path'=>$path]]);
        $rec->ocr_text = is_string($text) ? $text : null;
        $rec->save();
      } catch (\Throwable $ex) {}
    }
    return redirect()->back()->with('ok','Receipt uploaded');
  }
}
