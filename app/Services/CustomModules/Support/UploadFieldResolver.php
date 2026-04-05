<?php
namespace App\Services\CustomModules\Support;

use Illuminate\Http\Request;
use Illuminate\Http\UploadedFile;

class UploadFieldResolver
{
    public function fromRequest(Request $request): ?UploadedFile
    {
        return $request->file('file') ?? $request->file('download_file') ?? $request->file('module') ?? $request->file('zip');
    }
}
