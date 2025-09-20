<?php namespace Elseyyid\LaravelJsonLocationsManager\Models;

use Illuminate\Database\Eloquent\Model;

class Strings extends Model
{
    protected $connection = 'mysql';
    protected $table = 'strings';
    protected $primaryKey = 'code';

    protected $guarded = [];
}
