<?php
use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration {
  public function up(): void {
    if (!Schema::hasTable('expense_receipts')) {
      Schema::create('expense_receipts', function (Blueprint $t) {
        $t->id();
        $t->unsignedBigInteger('expense_id');
        $t->string('path');
        $t->string('mime')->nullable();
        $t->unsignedBigInteger('size')->default(0);
        $t->text('ocr_text')->nullable();
        $t->timestamps();
      });
    }
  }
  public function down(): void { Schema::dropIfExists('expense_receipts'); }
};
