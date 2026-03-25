<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

class CreateAccountsTransactionsTable extends Migration
{
    /**
     * Run the migrations.
     *
     * @return void
     */
    public function up()
    {
        Schema::create('accounting_accounts_transactions', function (Blueprint $table) {
            $table->bigIncrements('id');
            $table->unsignedBigInteger('accounting_account_id');
            $table->integer('acc_trans_mapping_id')->nullable()->comment('id form accounting_acc_trans_mapping table');
            $table->integer('transaction_id')->nullable()->comment('id form transactions table');
            $table->integer('transaction_payment_id')->nullable()->comment('id form transaction_payments table');
            $table->decimal('amount', 22, 4);
            $table->string('type', 100)->comment('debit, credit etc');
            $table->string('sub_type', 100);
            $table->string('map_type', 100)->nullable();
            $table->integer('created_by');
            $table->dateTime('operation_date');
            $table->text('note')->nullable();
            $table->timestamps();
        });
    }

    /**
     * Reverse the migrations.
     *
     * @return void
     */
    public function down()
    {
        Schema::dropIfExists('accounting_accounts_transactions');
    }
}
