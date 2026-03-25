<?php

/**
 * Author: Emmanuel Paul Mnzava
 * Twitter: @epmnzava
 * Github:https://github.com/dbrax/income-expense
 * Email: epmnzava@gmail.com
 * 
 */

namespace  Epmnzava\IncomeExpense\Models;

use Illuminate\Database\Eloquent\Model;
use Epmnzava\IncomeExpense\Models\IncomeCategory;


class Income extends Model
{

    protected $guarded = [];
    protected $table="income";

       public function category()
    {
        return $this->belongsTo(IncomeCategory::class,'incomecategory');
    }


}
