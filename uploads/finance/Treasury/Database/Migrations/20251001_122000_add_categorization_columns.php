<?php
use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;
return new class extends Migration {
  public function up(): void {
    if (Schema::hasTable('bank_transactions')) {
      Schema::table('bank_transactions', function(Blueprint $t){
        if (!Schema::hasColumn('bank_transactions','account_code')) $t->string('account_code')->nullable()->after('status');
        if (!Schema::hasColumn('bank_transactions','category')) $t->string('category')->nullable()->after('account_code');
      });
    }
  }
  public function down(): void {
    if (Schema::hasTable('bank_transactions')) {
      Schema::table('bank_transactions', function(Blueprint $t){
        if (Schema::hasColumn('bank_transactions','category')) $t->dropColumn('category');
        if (Schema::hasColumn('bank_transactions','account_code')) $t->dropColumn('account_code');
      });
    }
  }
};
