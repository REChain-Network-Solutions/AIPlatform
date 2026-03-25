<?php
use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration {
  public function up(): void {
    if (Schema::hasTable('expense')) {
      Schema::table('expense', function (Blueprint $t) {
        if (!Schema::hasColumn('expense','status')) $t->string('status')->default('draft')->after('description');
        if (!Schema::hasColumn('expense','approved_by')) $t->unsignedBigInteger('approved_by')->nullable()->after('status');
        if (!Schema::hasColumn('expense','approved_at')) $t->timestamp('approved_at')->nullable()->after('approved_by');
        if (!Schema::hasColumn('expense','reimbursed_at')) $t->timestamp('reimbursed_at')->nullable()->after('approved_at');
      });
    }
  }
  public function down(): void {
    if (Schema::hasTable('expense')) {
      Schema::table('expense', function (Blueprint $t) {
        if (Schema::hasColumn('expense','reimbursed_at')) $t->dropColumn('reimbursed_at');
        if (Schema::hasColumn('expense','approved_at')) $t->dropColumn('approved_at');
        if (Schema::hasColumn('expense','approved_by')) $t->dropColumn('approved_by');
        if (Schema::hasColumn('expense','status')) $t->dropColumn('status');
      });
    }
  }
};
