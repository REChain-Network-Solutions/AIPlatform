# YooKassa API PHP Validator

[![Latest Stable Version](https://img.shields.io/packagist/v/yoomoney/yookassa-sdk-validator?label=stable)](https://packagist.org/packages/yoomoney/yookassa-sdk-validator)
[![Total Downloads](https://img.shields.io/packagist/dt/yoomoney/yookassa-sdk-validator)](https://packagist.org/packages/yoomoney/yookassa-sdk-validator)
[![Monthly Downloads](https://img.shields.io/packagist/dm/yoomoney/yookassa-sdk-validator)](https://packagist.org/packages/yoomoney/yookassa-sdk-validator)
[![License](https://img.shields.io/packagist/l/yoomoney/yookassa-sdk-validator)](https://packagist.org/packages/yoomoney/yookassa-sdk-validator)

Библиотека для валидирования значений, присваиваемых полям объекта, через чтение атрибутов этих полей.
Предназначена для использования в составе [YooKassa API PHP Client Library](https://git.yoomoney.ru/projects/SDK/repos/yookassa-sdk-php/browse)

## Требования
PHP 8.0 (и выше)

## Установка
### В консоли с помощью Composer

1. [Установите менеджер пакетов Composer](https://getcomposer.org/download/).
2. В консоли выполните команду:
```bash
composer require yoomoney/yookassa-sdk-validator
```

### В файле composer.json своего проекта
1. Добавьте строку `"yoomoney/yookassa-sdk-validator": "^1.0"` в список зависимостей вашего проекта в файле composer.json:
```
...
    "require": {
        "php": ">=8.0",
        "yoomoney/yookassa-sdk-validator": "^1.0"
...
```
2. Обновите зависимости проекта. В консоли перейдите в каталог, где лежит composer.json, и выполните команду:
```bash
composer install
```
3. В коде вашего проекта подключите автозагрузку файлов валидатора:
```php
require __DIR__ . '/vendor/autoload.php';
```

## Начало работы

1. Импортируйте нужные классы валидатора:
```php
use YooKassa\Validator\Validator;
use YooKassa\Validator\Constraints as Assert;
```
2. Добавьте нужные правила для валидации полей класса через атрибуты:
```php
#[Assert\NotBlank]
#[Assert\Length(min: 2)]
private string $title;
```
3. Создайте экземпляр валидатора, передав в конструктор экземпляр класса, поля которого необходимо валидировать:
```php
$validator = new Validator($this);
```
4. Вызовите функцию validatePropertyValue(), передав в нее название валидируемого поля и значение:
```php
$validator->validatePropertyValue('title', $title);
```
5. Если значение не будет соответствовать правилам, заданным через атрибуты, валидатор выбросит исключение.

Чтобы пропустить проверку по какому-либо правилу или списку правил, заданному для поля класса, передайте массив с именами классов-правил в качестве параметра в функцию validatePropertyValue:
```php
$validator->validatePropertyValue('title', $title, [Assert\Length::class]);
```

Чтобы получить список правил для конкретного поля, вызовите функцию getRulesByPropName(), передав в качестве параметра название поля:
```php
$constraintsList = $validator->getRulesByPropName('title');
```

## Example
```php
<?php

require_once './vendor/autoload.php';

use YooKassa\Validator\Constraints as Assert;
use YooKassa\Validator\Validator;

class PaymentItemModel
{
    #[Assert\Length(min: 5)]
    private ?string $title;

    private Validator $validator;

    public function __construct()
    {
        $this->validator = new Validator($this);
    }

    public function setTitle(?string $title): PaymentItemModel
    {
        $this->validator->validatePropertyValue('title', $title);
        $this->title = $title;
        return $this;
    }
}

$paymentItem = new PaymentItemModel();
try {
    // Валидатор не выбросит исключение
    $paymentItem->setTitle('title');
    echo 'success!';
} catch (Exception $exception) {
    var_dump($exception->getMessage());
}

try {
    // Валидатор выбросит исключение
    $paymentItem->setTitle('titl');
} catch (Exception $exception) {
    echo 'fail!';
    var_dump($exception->getMessage());
}
```