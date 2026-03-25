<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        Schema::table('quotes', function (Blueprint $table) {
            if (!Schema::hasColumn('quotes', 'sent_at')) $table->timestamp('sent_at')->nullable()->after('status');
            if (!Schema::hasColumn('quotes', 'accepted_at')) $table->timestamp('accepted_at')->nullable()->after('sent_at');
            if (!Schema::hasColumn('quotes', 'rejected_at')) $table->timestamp('rejected_at')->nullable()->after('accepted_at');
        });
    }

    public function down(): void
    {
        Schema::table('quotes', function (Blueprint $table) {
            if (Schema::hasColumn('quotes', 'sent_at')) $table->dropColumn('sent_at');
            if (Schema::hasColumn('quotes', 'accepted_at')) $table->dropColumn('accepted_at');
            if (Schema::hasColumn('quotes', 'rejected_at')) $table->dropColumn('rejected_at');
        });
    }
};
