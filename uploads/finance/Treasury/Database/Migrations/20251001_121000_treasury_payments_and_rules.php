<?php
use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;
return new class extends Migration {
  public function up(): void {
    if (!Schema::hasTable('payment_runs')) {
      Schema::create('payment_runs', function (Blueprint $t) {
        $t->id();
        $t->date('scheduled_on');
        $t->string('status')->default('draft');
        $t->unsignedBigInteger('bank_account_id')->nullable();
        $t->string('posted_journal_id')->nullable();
        $t->timestamps();
      });
    }
    if (!Schema::hasTable('payment_lines')) {
      Schema::create('payment_lines', function (Blueprint $t) {
        $t->id();
        $t->unsignedBigInteger('payment_run_id');
        $t->string('beneficiary');
        $t->decimal('amount',14,2);
        $t->string('reference')->nullable();
        $t->string('status')->default('pending');
        $t->timestamps();
      });
    }
    if (!Schema::hasTable('reconciliation_rules')) {
      Schema::create('reconciliation_rules', function (Blueprint $t) {
        $t->id();
        $t->string('pattern'); // simple LIKE match against description
        $t->string('account_code')->nullable();
        $t->enum('direction', ['in','out'])->default('out');
        $t->timestamps();
      });
    }
  }
  public function down(): void {
    Schema::dropIfExists('payment_lines');
    Schema::dropIfExists('payment_runs');
    Schema::dropIfExists('reconciliation_rules');
  }
};
