<?php

use Illuminate\Database\Migrations\Migration;
use Illuminate\Database\Schema\Blueprint;
use Illuminate\Support\Facades\Schema;

class CreateAccountingBudgetsTable extends Migration
{
    /**
     * Run the migrations.
     *
     * @return void
     */
    public function up()
    {
        Schema::create('accounting_budgets', function (Blueprint $table) {
            $table->bigIncrements('id');
            $table->unsignedBigInteger('accounting_account_id');
            $table->integer('financial_year');
            $table->decimal('jan', 22, 4)->nullable();
            $table->decimal('feb', 22, 4)->nullable();
            $table->decimal('mar', 22, 4)->nullable();
            $table->decimal('apr', 22, 4)->nullable();
            $table->decimal('may', 22, 4)->nullable();
            $table->decimal('jun', 22, 4)->nullable();
            $table->decimal('jul', 22, 4)->nullable();
            $table->decimal('aug', 22, 4)->nullable();
            $table->decimal('sep', 22, 4)->nullable();
            $table->decimal('oct', 22, 4)->nullable();
            $table->decimal('nov', 22, 4)->nullable();
            $table->decimal('dec', 22, 4)->nullable();
            $table->decimal('quarter_1', 22, 4)->nullable();
            $table->decimal('quarter_2', 22, 4)->nullable();
            $table->decimal('quarter_3', 22, 4)->nullable();
            $table->decimal('quarter_4', 22, 4)->nullable();
            $table->decimal('yearly', 22, 4)->nullable();
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
        Schema::dropIfExists('accounting_budgets');
    }
}
