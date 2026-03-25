<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

return new class extends Migration
{
    public function up(): void
    {
        if (!Schema::hasTable('einvoice_invoices')) {
            Schema::create('einvoice_invoices', function (Blueprint $table) {
                $table->bigIncrements('id');
                $table->unsignedBigInteger('client_id')->nullable();
                $table->string('currency', 8)->default('USD');
                $table->string('status', 20)->default('draft'); // draft, sent, paid, void
                $table->date('due_date')->nullable();
                $table->text('notes')->nullable();
                $table->decimal('subtotal', 12, 2)->default(0);
                $table->decimal('tax_total', 12, 2)->default(0);
                $table->decimal('grand_total', 12, 2)->default(0);
                $table->timestamps();
                $table->index(['client_id', 'status']);
            });
        }
    }

    public function down(): void
    {
        Schema::dropIfExists('einvoice_invoices');
    }
};
