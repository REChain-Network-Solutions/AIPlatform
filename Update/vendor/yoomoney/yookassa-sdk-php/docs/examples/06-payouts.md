## Работа с выплатами

Выплата — это сумма денег, которую вы переводите физическому лицу или самозанятому. С помощью API вы можете создать выплату и получить о ней актуальную информацию.

Выплаты используются в следующих платежных решениях ЮKassa:

* Выплаты — вы как компания переводите деньги физическим лицам и самозанятым (например, выплачиваете кэшбэк пользователям).
* Безопасная сделка — ваша платформа в рамках созданной сделки переводит оплату от одного физического лица другому.

SDK позволяет проводить выплаты, а также получать информацию о них.

Объект выплаты `PayoutResponse` содержит всю информацию о выплате, актуальную на текущий момент времени. Объект формируется при создании выплаты и приходит в ответ на любой запрос, связанный с выплатами.

Набор возвращаемых параметров зависит от статуса объекта (значение параметра status) и того, какие параметры вы передали в запросе на создание выплаты.

* [Запрос на выплату продавцу](#Запрос-на-выплату-продавцу)
  * [Проведение выплаты на банковскую карту](#Проведение-выплаты-на-банковскую-карту)
  * [Проведение выплаты в кошелек ЮMoney](#Проведение-выплаты-в-кошелек-юmoney)
  * [Проведение выплаты через СБП](#Проведение-выплаты-через-сбп)
  * [Выплаты самозанятым](#Выплаты-самозанятым)
  * [Проведение выплаты по безопасной сделке](#Проведение-выплаты-по-безопасной-сделке)
* [Получить информацию о выплате](#Получить-информацию-о-выплате)

---

### Запрос на выплату продавцу <a name="Запрос-на-выплату-продавцу"></a>

[Выплата продавцу в документации](https://yookassa.ru/developers/api?lang=php#create_payout)

Запрос позволяет перечислить продавцу оплату за выполненную услугу или проданный товар в рамках «Безопасной сделки». Выплату можно сделать на банковскую карту или в кошелек ЮMoney.

В ответ на запрос придет объект выплаты — `PayoutResponse` — в актуальном статусе.

[Подробнее о проведении выплат](https://yookassa.ru/developers/solutions-for-platforms/safe-deal/integration/payouts)

#### Проведение выплаты на банковскую карту <a name="Проведение-выплаты-на-банковскую-карту"></a>

```php
require_once 'vendor/autoload.php';

$client = new \YooKassa\Client();
$client->setAuth('xxxxxx', 'test_XXXXXXX');

$request = [
    'amount' => [
        'value' => '80.00',
        'currency' => 'RUB',
    ],
    'payout_destination_data' => [
        'type' => 'bank_card',
        'card' => [
            'number' => '5555555555554477',
        ],
    ],
    'description' => 'Выплата по заказу №37',
    'metadata' => [
        'order_id' => '37',
    ],
];
$idempotenceKey = uniqid('', true);
try {
    $result = $client->createPayout($request, $idempotenceKey);
} catch (\Exception $e) {
    $result = $e;
}

var_dump($result);
```

#### Проведение выплаты в кошелек ЮMoney <a name="Проведение-выплаты-в-кошелек-юmoney"></a>

```php
require_once 'vendor/autoload.php';

$client = new \YooKassa\Client();
$client->setAuth('xxxxxx', 'test_XXXXXXX');

$request = [
    'amount' => [
        'value' => '80.00',
        'currency' => 'RUB',
    ],
    'payout_destination_data' => [
        'type' => 'yoo_money',
        'account_number' => '4100116075156746',
    ],
    'description' => 'Выплата по заказу №37',
    'metadata' => [
        'order_id' => '37',
    ],
];
$idempotenceKey = uniqid('', true);
try {
    $result = $client->createPayout($request, $idempotenceKey);
} catch (\Exception $e) {
    $result = $e;
}

var_dump($result);
```

#### Проведение выплаты через СБП <a name="Проведение-выплаты-через-сбп"></a>

```php
require_once 'vendor/autoload.php';

$client = new \YooKassa\Client();
$client->setAuth('xxxxxx', 'test_XXXXXXX');

$request = [
    'amount' => [
        'value' => '80.00',
        'currency' => 'RUB',
    ],
    'payout_destination_data' => [
        'type' => 'sbp',
        'phone' => '79000000000',
        'bank_id' => '100000000111',
    ],
    'description' => 'Выплата по заказу №37',
    'metadata' => [
        'order_id' => '37',
    ],
];
$idempotenceKey = uniqid('', true);
try {
    $result = $client->createPayout($request, $idempotenceKey);
} catch (\Exception $e) {
    $result = $e;
}

var_dump($result);
```

#### Выплаты самозанятым <a name="Выплаты-самозанятым"></a>

```php
require_once 'vendor/autoload.php';

$client = new \YooKassa\Client();
$client->setAuth('xxxxxx', 'test_XXXXXXX');

$request = [
    'amount' => [
        'value' => '80.00',
        'currency' => 'RUB',
    ],
    'payout_token' => '<Синоним банковской карты>',
    'self_employed' => [
        'id' => 'se-d6b9b3fa-0cb8-4aa8-b3c0-254bf0358d4c',
    ],
    'receipt_data' => [
        'service_name' => 'Доставка документов'
    ],
    'description' => 'Выплата по заказу №37',
    'metadata' => [
        'order_id' => '37',
        'courier_id' => '001',
    ],
];
$idempotenceKey = uniqid('', true);
try {
    $result = $client->createPayout($request, $idempotenceKey);
} catch (\Exception $e) {
    $result = $e;
}

var_dump($result);
```

#### Проведение выплаты по безопасной сделке <a name="Проведение-выплаты-по-безопасной-сделке"></a>

```php
require_once 'vendor/autoload.php';

$client = new \YooKassa\Client();
$client->setAuth('xxxxxx', 'test_XXXXXXX');

$request = [
    'amount' => [
        'value' => '800.00',
        'currency' => 'RUB',
    ],
    'payout_token' => '<Синоним банковской карты>',
    'description' => 'Выплата по заказу №37',
    'metadata' => [
        'order_id' => '37'
    ],
    'deal' => [
        'id' => 'dl-285e5ee7-0022-5000-8000-01516a44b147',
    ],
];
$idempotenceKey = uniqid('', true);
try {
    $result = $client->createPayout($request, $idempotenceKey);
} catch (\Exception $e) {
    $result = $e;
}

var_dump($result);
```

---

### Получить информацию о выплате <a name="Получить-информацию-о-выплате"></a>

[Информация о выплате в документации](https://yookassa.ru/developers/api?lang=php#get_payout)

Запрос позволяет получить информацию о текущем состоянии выплаты по ее уникальному идентификатору.

[Данные для аутентификации запросов](https://yookassa.ru/developers/using-api/interaction-format#auth) зависят от того, какое платежное решение вы используете — [обычные выплаты](https://yookassa.ru/developers/payouts/overview) или выплаты в рамках «[Безопасной сделки](https://yookassa.ru/developers/solutions-for-platforms/safe-deal/basics)».

В ответ на запрос придет объект выплаты — `PayoutResponse` — в актуальном статусе.

```php
require_once 'vendor/autoload.php';

$client = new \YooKassa\Client();
$client->setAuth('xxxxxx', 'test_XXXXXXX');

$payoutId = 'po-285c0ab7-0003-5000-9000-0e1166498fda';
try {
    $response = $client->getPayoutInfo($payoutId);
} catch (\Exception $e) {
    $response = $e;
}

var_dump($response);
```
