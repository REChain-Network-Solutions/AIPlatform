<?php
namespace Modules\CustomerFeedback\Http\Controllers;
use Illuminate\Support\Facades\Storage; use App\Http\Controllers\AccountBaseController; use Modules\CustomerFeedback\Entities\FeedbackFile;
class FeedbackFileController extends AccountBaseController {
 public function show(FeedbackFile $file){ abort_unless(Storage::disk('public')->exists($file->file_path),404); return response()->file(Storage::disk('public')->path($file->file_path)); }
 public function download(FeedbackFile $file){ abort_unless(Storage::disk('public')->exists($file->file_path),404); return Storage::disk('public')->download($file->file_path,$file->filename); }
 public function destroy(FeedbackFile $file){ if(Storage::disk('public')->exists($file->file_path)){ Storage::disk('public')->delete($file->file_path);} $file->delete(); return redirect()->back()->with('success','File deleted.'); }
}
