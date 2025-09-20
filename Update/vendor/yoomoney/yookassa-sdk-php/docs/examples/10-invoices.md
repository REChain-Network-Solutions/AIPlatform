## Работа со счетами

С помощью SDK можно выставлять счета на оплату. Счет — это страница ЮKassa, на которой пользователь увидит описание заказа и сможет заплатить в любой удобный момент в течение заданного вами срока. [Выставление счетов](https://yookassa.ru/developers/payment-acceptance/scenario-extensions/invoices/basics)

* [Запрос на создание счета](#Запрос-на-создание-счета)
* [Запрос на создание счета через билдер](#Запрос-на-создание-счета-через-билдер)
* [Получить информацию о счете](#Получить-информацию-о-счете)

---

### Запрос на создание счета <a name="Запрос-на-создание-счета"></a>

[Создание счета в документации](https://yookassa.ru/developers/api?codeLang=php#create_invoice)

Используйте этот запрос, чтобы создать в ЮKassa [объект счета](https://yookassa.ru/developers/api?codeLang=bash#invoice_object). В запросе необходимо передать данные о заказе, которые отобразятся на странице счета, и данные для проведения платежа.

В ответ на запрос придет объект счета — `InvoiceResponse` — в актуальном статусе.

```php
require_once 'vendor/autoload.php';

$client = new \YooKassa\Client();
$client->setAuth('xxxxxx', 'test_XXXXXXX');

try {
    $response = $client->createInvoice(
        [
            'payment_data' => [
                'amount' => [
                    'value' => '10.00',
                    'currency' => 'RUB',
                ],
                'capture' => true,
                'description' => 'Заказ №37',
                'metadata' => [
                    'order_id' => '37',
                ],
            ],
            'cart' => [
                [
                    'description' => 'Товар арт. 12345',
                    'price' => [
                        'value' => '9.00',
                        'currency' => 'RUB',
                    ],
                    'discount_price' => [
                        'value' => '7.00',
                        'currency' => 'RUB',
                    ],
                    'quantity' => 1.000,
                ],
                [
                    'description' => 'Товар арт. 67890',
                    'price' => [
                        'value' => '1.00',
                        'currency' => 'RUB',
                    ],
                    'quantity' => 3.000,
                ],
            ],
            'delivery_method_data' => [
                'type' => 'self',
            ],
            'locale' => 'ru_RU',
            'expires_at' => '2024-10-18T10:51:18.139Z',
            'description' => 'Счет на оплату заказа номер 37',
            'metadata' => [
                'order_id' => '37',
            ],
        ],
        uniqid('', true)
    );
    if ($response->getDeliveryMethod() && $response->getDeliveryMethod()->getType() === DeliveryMethodType::SELF) {
        echo $response->getDeliveryMethod()->getUrl(); // Получаем ссылку на счёт и передаем плательщику
    }
} catch (\Exception $e) {
    var_dump($response);
}
```

---

### Запрос на создание счета через билдер <a name="Запрос-на-создание-счета-через-билдер"></a>

[Информация о создании счета в документации](https://yookassa.ru/developers/api?codeLang=php#create_invoice)

Билдер позволяет создать объект счета — `CreateInvoiceRequest` — программным способом, через объекты.

```php
require_once 'vendor/autoload.php';

$client = new \YooKassa\Client();
$client->setAuth('xxxxxx', 'test_XXXXXXX');

try {
    $invoiceBuilder = \YooKassa\Request\Invoices\CreateInvoiceRequest::builder();
    $invoiceBuilder
        ->setPaymentData([
            'amount' => [
                'value' => '10.00',
                'currency' => 'RUB',
            ],
            'capture' => true,
            'description' => 'Заказ №37',
            'metadata' => [
                'order_id' => '37',
            ],
        ])
        ->addCartItem([
            'description' => 'Товар арт. 12345',
            'price' => [
                'value' => '9.00',
                'currency' => 'RUB',
            ],
            'discount_price' => [
                'value' => '7.00',
                'currency' => 'RUB',
            ],
            'quantity' => 1.000,
        ])
        ->addCartItem([
            'description' => 'Товар арт. 67890',
            'price' => [
                'value' => '1.00',
                'currency' => 'RUB',
            ],
            'quantity' => 3.000,
        ])
        ->setLocale('ru_RU')
        ->setExpiresAt('2024-10-18T10:51:18.139Z')
        ->setDeliveryMethodData(new \YooKassa\Request\Invoices\DeliveryMethodData\DeliveryMethodDataSelf())
        ->setDescription('Счет на оплату заказа номер #37')
        ->setMetadata([
            'order_id' => '37',
        ]);

    // Создаем объект запроса
    $request = $invoiceBuilder->build();

    // Можно изменить данные, если нужно
    $request->setDescription('Счет на оплату заказа номер 37');

    $idempotenceKey = uniqid('', true);
    $response = $client->createInvoice($request, $idempotenceKey);
    if ($response->getDeliveryMethod() && $response->getDeliveryMethod()->getType() === DeliveryMethodType::SELF) {
        echo $response->getDeliveryMethod()->getUrl(); // Получаем ссылку на счёт и передаем плательщику
    }
} catch (Exception $e) {
    var_dump($e);
}
```

---

### Получить информацию о счете <a name="Получить-информацию-о-счете"></a>

[Информация о счете в документации](https://yookassa.ru/developers/api?codeLang=php#get_invoice)

Запрос позволяет получить информацию о текущем состоянии счета по его уникальному идентификатору.

В ответ на запрос придет объект счета — `InvoiceResponse` — в актуальном статусе.

```php
require_once 'vendor/autoload.php';

$client = new \YooKassa\Client();
$client->setAuth('xxxxxx', 'test_XXXXXXX');

try {
    $invoiceId = 'in-e44e8088-bd73-43b1-959a-954f3a7d0c54';
    $response = $client->getInvoiceInfo($invoiceId);
    print_r($response->toArray());
    if ($response->getDeliveryMethod()) { // Если счет не оплачен
        if ($response->getDeliveryMethod()->getType() === DeliveryMethodType::SELF) {
            echo $response->getDeliveryMethod()->getUrl(); // Получаем ссылку на счёт и передаем плательщику
        }
    }
    if ($response->getPaymentDetails()) { // Если счет оплачен
        echo $response->getPaymentDetails()->getStatus();
        $paymentResponse = $client->getPaymentInfo($response->getPaymentDetails()->getId());
        echo $paymentResponse->getStatus();
    }

} catch (Exception $e) {
    var_dump($e);
}
```
