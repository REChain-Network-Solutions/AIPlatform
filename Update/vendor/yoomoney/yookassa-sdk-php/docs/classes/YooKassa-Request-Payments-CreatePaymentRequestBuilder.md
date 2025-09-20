# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Request\Payments\CreatePaymentRequestBuilder
### Namespace: [\YooKassa\Request\Payments](../namespaces/yookassa-request-payments.md)
---
**Summary:**

Класс, представляющий модель CreatePaymentRequestBuilder.

**Description:**

Класс билдера объекта запроса на создание платежа, передаваемого в методы клиента API.

---
### Examples
Пример использования билдера

```php
try {
    $builder = \YooKassa\Request\Payments\CreatePaymentRequest::builder();
    $builder->setAmount(100)
        ->setCurrency(\YooKassa\Model\CurrencyCode::RUB)
        ->setCapture(true)
        ->setDescription('Оплата заказа 112233')
        ->setMetadata([
            'cms_name' => 'yoo_api_test',
            'order_id' => '112233',
            'language' => 'ru',
            'transaction_id' => '123-456-789',
        ])
    ;

    // Устанавливаем страницу для редиректа после оплаты
    $builder->setConfirmation([
        'type' => \YooKassa\Model\Payment\ConfirmationType::REDIRECT,
        'returnUrl' => 'https://merchant-site.ru/payment-return-page',
    ]);

    // Можем установить конкретный способ оплаты
    $builder->setPaymentMethodData(\YooKassa\Model\Payment\PaymentMethodType::BANK_CARD);

    // Составляем чек
    $builder->setReceiptEmail('john.doe@merchant.com');
    $builder->setReceiptPhone('71111111111');
    // Добавим товар
    $builder->addReceiptItem(
        'Платок Gucci',
        3000,
        1.0,
        2,
        'full_payment',
        'commodity'
    );
    // Добавим доставку
    $builder->addReceiptShipping(
        'Delivery/Shipping/Доставка',
        100,
        1,
        \YooKassa\Model\Receipt\PaymentMode::FULL_PAYMENT,
        \YooKassa\Model\Receipt\PaymentSubject::SERVICE
    );

    // Можно добавить распределение денег по магазинам
    $builder->setTransfers([
        [
            'account_id' => '1b68e7b15f3f',
            'amount' => [
                'value' => 1000,
                'currency' => \YooKassa\Model\CurrencyCode::RUB,
            ],
        ],
        [
            'account_id' => '0c37205b3208',
            'amount' => [
                'value' => 2000,
                'currency' => \YooKassa\Model\CurrencyCode::RUB,
            ],
        ],
    ]);

    // Создаем объект запроса
    $request = $builder->build();

    // Можно изменить данные, если нужно
    $request->setDescription($request->getDescription() . ' - merchant comment');

    $idempotenceKey = uniqid('', true);
    $response = $client->createPayment($request, $idempotenceKey);
} catch (\Exception $e) {
    $response = $e;
}

var_dump($response);

```

---
### Constants
* No constants found

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| protected | [$amount](../classes/YooKassa-Request-Payments-AbstractPaymentRequestBuilder.md#property_amount) |  | Сумма |
| protected | [$currentObject](../classes/YooKassa-Request-Payments-CreatePaymentRequestBuilder.md#property_currentObject) |  | Собираемый объект запроса. |
| protected | [$receipt](../classes/YooKassa-Request-Payments-AbstractPaymentRequestBuilder.md#property_receipt) |  | Объект с информацией о чеке |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [__construct()](../classes/YooKassa-Common-AbstractRequestBuilder.md#method___construct) |  | Конструктор, инициализирует пустой запрос, который в будущем начнём собирать. |
| public | [addReceiptItem()](../classes/YooKassa-Request-Payments-AbstractPaymentRequestBuilder.md#method_addReceiptItem) |  | Добавляет в чек товар |
| public | [addReceiptShipping()](../classes/YooKassa-Request-Payments-AbstractPaymentRequestBuilder.md#method_addReceiptShipping) |  | Добавляет в чек доставку товара. |
| public | [addTransfer()](../classes/YooKassa-Request-Payments-AbstractPaymentRequestBuilder.md#method_addTransfer) |  | Добавляет трансфер. |
| public | [build()](../classes/YooKassa-Request-Payments-CreatePaymentRequestBuilder.md#method_build) |  | Строит и возвращает объект запроса для отправки в API ЮKassa. |
| public | [setAccountId()](../classes/YooKassa-Request-Payments-CreatePaymentRequestBuilder.md#method_setAccountId) | *deprecated* | Устанавливает идентификатор магазина получателя платежа. |
| public | [setAirline()](../classes/YooKassa-Request-Payments-CreatePaymentRequestBuilder.md#method_setAirline) |  | Устанавливает информацию об авиабилетах. |
| public | [setAmount()](../classes/YooKassa-Request-Payments-AbstractPaymentRequestBuilder.md#method_setAmount) |  | Устанавливает сумму. |
| public | [setCapture()](../classes/YooKassa-Request-Payments-CreatePaymentRequestBuilder.md#method_setCapture) |  | Устанавливает флаг автоматического принятия поступившей оплаты. |
| public | [setClientIp()](../classes/YooKassa-Request-Payments-CreatePaymentRequestBuilder.md#method_setClientIp) |  | Устанавливает IP адрес покупателя. |
| public | [setConfirmation()](../classes/YooKassa-Request-Payments-CreatePaymentRequestBuilder.md#method_setConfirmation) |  | Устанавливает способ подтверждения платежа. |
| public | [setCurrency()](../classes/YooKassa-Request-Payments-AbstractPaymentRequestBuilder.md#method_setCurrency) |  | Устанавливает валюту в которой будет происходить подтверждение оплаты заказа. |
| public | [setDeal()](../classes/YooKassa-Request-Payments-CreatePaymentRequestBuilder.md#method_setDeal) |  | Устанавливает сделку. |
| public | [setDescription()](../classes/YooKassa-Request-Payments-CreatePaymentRequestBuilder.md#method_setDescription) |  | Устанавливает описание транзакции. |
| public | [setFraudData()](../classes/YooKassa-Request-Payments-CreatePaymentRequestBuilder.md#method_setFraudData) | *deprecated* | Устанавливает информацию для проверки операции на мошенничество. |
| public | [setGatewayId()](../classes/YooKassa-Request-Payments-CreatePaymentRequestBuilder.md#method_setGatewayId) |  | Устанавливает идентификатор шлюза. |
| public | [setMerchantCustomerId()](../classes/YooKassa-Request-Payments-CreatePaymentRequestBuilder.md#method_setMerchantCustomerId) |  | Устанавливает идентификатор покупателя в вашей системе. |
| public | [setMetadata()](../classes/YooKassa-Request-Payments-CreatePaymentRequestBuilder.md#method_setMetadata) |  | Устанавливает метаданные, привязанные к платежу. |
| public | [setOptions()](../classes/YooKassa-Common-AbstractRequestBuilder.md#method_setOptions) |  | Устанавливает свойства запроса из массива. |
| public | [setPaymentMethodData()](../classes/YooKassa-Request-Payments-CreatePaymentRequestBuilder.md#method_setPaymentMethodData) |  | Устанавливает объект с информацией для создания метода оплаты. |
| public | [setPaymentMethodId()](../classes/YooKassa-Request-Payments-CreatePaymentRequestBuilder.md#method_setPaymentMethodId) |  | Устанавливает идентификатор записи о сохранённых данных покупателя. |
| public | [setPaymentOrder()](../classes/YooKassa-Request-Payments-CreatePaymentRequestBuilder.md#method_setPaymentOrder) |  | Устанавливает платежное поручение. |
| public | [setPaymentToken()](../classes/YooKassa-Request-Payments-CreatePaymentRequestBuilder.md#method_setPaymentToken) |  | Устанавливает одноразовый токен для проведения оплаты. |
| public | [setReceipt()](../classes/YooKassa-Request-Payments-AbstractPaymentRequestBuilder.md#method_setReceipt) |  | Устанавливает чек. |
| public | [setReceiptEmail()](../classes/YooKassa-Request-Payments-AbstractPaymentRequestBuilder.md#method_setReceiptEmail) |  | Устанавливает адрес электронной почты получателя чека. |
| public | [setReceiptIndustryDetails()](../classes/YooKassa-Request-Payments-CreatePaymentRequestBuilder.md#method_setReceiptIndustryDetails) |  | Устанавливает отраслевой реквизит чека. |
| public | [setReceiptItems()](../classes/YooKassa-Request-Payments-AbstractPaymentRequestBuilder.md#method_setReceiptItems) |  | Устанавливает список товаров для создания чека. |
| public | [setReceiptOperationalDetails()](../classes/YooKassa-Request-Payments-CreatePaymentRequestBuilder.md#method_setReceiptOperationalDetails) |  | Устанавливает отраслевой реквизит чека. |
| public | [setReceiptPhone()](../classes/YooKassa-Request-Payments-AbstractPaymentRequestBuilder.md#method_setReceiptPhone) |  | Устанавливает телефон получателя чека. |
| public | [setReceiver()](../classes/YooKassa-Request-Payments-CreatePaymentRequestBuilder.md#method_setReceiver) |  | Устанавливает реквизиты получателя оплаты. |
| public | [setRecipient()](../classes/YooKassa-Request-Payments-CreatePaymentRequestBuilder.md#method_setRecipient) |  | Устанавливает получателя платежа из объекта или ассоциативного массива. |
| public | [setSavePaymentMethod()](../classes/YooKassa-Request-Payments-CreatePaymentRequestBuilder.md#method_setSavePaymentMethod) |  | Устанавливает флаг сохранения платёжных данных. Значение true инициирует создание многоразового payment_method. |
| public | [setTaxSystemCode()](../classes/YooKassa-Request-Payments-AbstractPaymentRequestBuilder.md#method_setTaxSystemCode) |  | Устанавливает код системы налогообложения. |
| public | [setTransfers()](../classes/YooKassa-Request-Payments-AbstractPaymentRequestBuilder.md#method_setTransfers) |  | Устанавливает трансферы. |
| protected | [getPaymentDataFactory()](../classes/YooKassa-Request-Payments-CreatePaymentRequestBuilder.md#method_getPaymentDataFactory) |  | Возвращает фабрику методов проведения платежей. |
| protected | [initCurrentObject()](../classes/YooKassa-Request-Payments-CreatePaymentRequestBuilder.md#method_initCurrentObject) |  | Инициализирует объект запроса, который в дальнейшем будет собираться билдером |

---
### Details
* File: [lib/Request/Payments/CreatePaymentRequestBuilder.php](../../lib/Request/Payments/CreatePaymentRequestBuilder.php)
* Package: YooKassa\Request
* Class Hierarchy:  
  * [\YooKassa\Common\AbstractRequestBuilder](../classes/YooKassa-Common-AbstractRequestBuilder.md)
  * [\YooKassa\Request\Payments\AbstractPaymentRequestBuilder](../classes/YooKassa-Request-Payments-AbstractPaymentRequestBuilder.md)
  * \YooKassa\Request\Payments\CreatePaymentRequestBuilder

* See Also:
  * [](https://yookassa.ru/developers/api)

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| category |  | Class |
| author |  | cms@yoomoney.ru |

---
## Properties
<a name="property_amount"></a>
#### protected $amount : ?\YooKassa\Model\MonetaryAmount
---
**Summary**

Сумма

**Type:** <a href="../?\YooKassa\Model\MonetaryAmount"><abbr title="?\YooKassa\Model\MonetaryAmount">MonetaryAmount</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Request\Payments\AbstractPaymentRequestBuilder](../classes/YooKassa-Request-Payments-AbstractPaymentRequestBuilder.md)


<a name="property_currentObject"></a>
#### protected $currentObject : ?\YooKassa\Common\AbstractRequestInterface
---
**Summary**

Собираемый объект запроса.

**Type:** <a href="../?\YooKassa\Common\AbstractRequestInterface"><abbr title="?\YooKassa\Common\AbstractRequestInterface">AbstractRequestInterface</abbr></a>

**Details:**


<a name="property_receipt"></a>
#### protected $receipt : ?\YooKassa\Model\Receipt\Receipt
---
**Summary**

Объект с информацией о чеке

**Type:** <a href="../?\YooKassa\Model\Receipt\Receipt"><abbr title="?\YooKassa\Model\Receipt\Receipt">Receipt</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Request\Payments\AbstractPaymentRequestBuilder](../classes/YooKassa-Request-Payments-AbstractPaymentRequestBuilder.md)



---
## Methods
<a name="method___construct" class="anchor"></a>
#### public __construct() : mixed

```php
public __construct() : mixed
```

**Summary**

Конструктор, инициализирует пустой запрос, который в будущем начнём собирать.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractRequestBuilder](../classes/YooKassa-Common-AbstractRequestBuilder.md)

**Returns:** mixed - 


<a name="method_addReceiptItem" class="anchor"></a>
#### public addReceiptItem() : self

```php
public addReceiptItem(string $title, string $price, float $quantity, int $vatCode, null|string $paymentMode = null, null|string $paymentSubject = null, null|mixed $productCode = null, null|mixed $countryOfOriginCode = null, null|mixed $customsDeclarationNumber = null, null|mixed $excise = null) : self
```

**Summary**

Добавляет в чек товар

**Details:**
* Inherited From: [\YooKassa\Request\Payments\AbstractPaymentRequestBuilder](../classes/YooKassa-Request-Payments-AbstractPaymentRequestBuilder.md)
* See Also:
 * [](\YooKassa\Request\Payments\PaymentSubject)
 * [](\YooKassa\Request\Payments\PaymentMode)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | title  | Название или описание товара |
| <code lang="php">string</code> | price  | Цена товара в валюте, заданной в заказе |
| <code lang="php">float</code> | quantity  | Количество товара |
| <code lang="php">int</code> | vatCode  | Ставка НДС |
| <code lang="php">null OR string</code> | paymentMode  | значение перечисления PaymentMode |
| <code lang="php">null OR string</code> | paymentSubject  | значение перечисления PaymentSubject |
| <code lang="php">null OR mixed</code> | productCode  |  |
| <code lang="php">null OR mixed</code> | countryOfOriginCode  |  |
| <code lang="php">null OR mixed</code> | customsDeclarationNumber  |  |
| <code lang="php">null OR mixed</code> | excise  |  |

**Returns:** self - Инстанс билдера запросов


<a name="method_addReceiptShipping" class="anchor"></a>
#### public addReceiptShipping() : self

```php
public addReceiptShipping(string $title, string $price, int $vatCode, null|string $paymentMode = null, null|string $paymentSubject = null) : self
```

**Summary**

Добавляет в чек доставку товара.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\AbstractPaymentRequestBuilder](../classes/YooKassa-Request-Payments-AbstractPaymentRequestBuilder.md)
* See Also:
 * [](\YooKassa\Request\Payments\PaymentSubject)
 * [](\YooKassa\Request\Payments\PaymentMode)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | title  | Название доставки в чеке |
| <code lang="php">string</code> | price  | Стоимость доставки |
| <code lang="php">int</code> | vatCode  | Ставка НДС |
| <code lang="php">null OR string</code> | paymentMode  | значение перечисления PaymentMode |
| <code lang="php">null OR string</code> | paymentSubject  | значение перечисления PaymentSubject |

**Returns:** self - Инстанс билдера запросов


<a name="method_addTransfer" class="anchor"></a>
#### public addTransfer() : self

```php
public addTransfer(array|\YooKassa\Request\Payments\TransferDataInterface $value) : self
```

**Summary**

Добавляет трансфер.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\AbstractPaymentRequestBuilder](../classes/YooKassa-Request-Payments-AbstractPaymentRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \YooKassa\Request\Payments\TransferDataInterface</code> | value  | Трансфер |

**Returns:** self - Инстанс билдера запросов


<a name="method_build" class="anchor"></a>
#### public build() : \YooKassa\Request\Payments\CreatePaymentRequestInterface|\YooKassa\Common\AbstractRequestInterface

```php
public build(null|array $options = null) : \YooKassa\Request\Payments\CreatePaymentRequestInterface|\YooKassa\Common\AbstractRequestInterface
```

**Summary**

Строит и возвращает объект запроса для отправки в API ЮKassa.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestBuilder](../classes/YooKassa-Request-Payments-CreatePaymentRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR array</code> | options  | Массив параметров для установки в объект запроса |

**Returns:** \YooKassa\Request\Payments\CreatePaymentRequestInterface|\YooKassa\Common\AbstractRequestInterface - Инстанс объекта запроса


<a name="method_setAccountId" class="anchor"></a>
#### (deprecated) - public setAccountId() : \YooKassa\Request\Payments\CreatePaymentRequestBuilder

```php
public setAccountId(string $value) : \YooKassa\Request\Payments\CreatePaymentRequestBuilder
```

**Summary**

Устанавливает идентификатор магазина получателя платежа.

**Deprecated**
DeprecatedБольше не используется
**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestBuilder](../classes/YooKassa-Request-Payments-CreatePaymentRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | value  | Идентификатор магазина |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\EmptyPropertyValueException | Выбрасывается если было передано пустое значение |
| \YooKassa\Common\Exceptions\InvalidPropertyValueTypeException | Выбрасывается если было передано не строковое значение |
| \Exception |  |

**Returns:** \YooKassa\Request\Payments\CreatePaymentRequestBuilder - Инстанс текущего билдера


<a name="method_setAirline" class="anchor"></a>
#### public setAirline() : \YooKassa\Request\Payments\CreatePaymentRequestBuilder

```php
public setAirline(\YooKassa\Request\Payments\AirlineInterface|array|null $value) : \YooKassa\Request\Payments\CreatePaymentRequestBuilder
```

**Summary**

Устанавливает информацию об авиабилетах.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestBuilder](../classes/YooKassa-Request-Payments-CreatePaymentRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Request\Payments\AirlineInterface OR array OR null</code> | value  | Объект данных длинной записи или ассоциативный массив с данными |

**Returns:** \YooKassa\Request\Payments\CreatePaymentRequestBuilder - 


<a name="method_setAmount" class="anchor"></a>
#### public setAmount() : self

```php
public setAmount(\YooKassa\Model\AmountInterface|array|\YooKassa\Request\Payments\numeric $value, string|null $currency = null) : self
```

**Summary**

Устанавливает сумму.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\AbstractPaymentRequestBuilder](../classes/YooKassa-Request-Payments-AbstractPaymentRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\AmountInterface OR array OR \YooKassa\Request\Payments\numeric</code> | value  | Сумма оплаты |
| <code lang="php">string OR null</code> | currency  | Валюта |

**Returns:** self - Инстанс билдера запросов


<a name="method_setCapture" class="anchor"></a>
#### public setCapture() : \YooKassa\Request\Payments\CreatePaymentRequestBuilder

```php
public setCapture(bool $value) : \YooKassa\Request\Payments\CreatePaymentRequestBuilder
```

**Summary**

Устанавливает флаг автоматического принятия поступившей оплаты.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestBuilder](../classes/YooKassa-Request-Payments-CreatePaymentRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">bool</code> | value  | Автоматически принять поступившую оплату |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\InvalidPropertyValueTypeException | Генерируется если переданный аргумент не кастится в bool |

**Returns:** \YooKassa\Request\Payments\CreatePaymentRequestBuilder - Инстанс текущего билдера


<a name="method_setClientIp" class="anchor"></a>
#### public setClientIp() : \YooKassa\Request\Payments\CreatePaymentRequestBuilder

```php
public setClientIp(string|null $value) : \YooKassa\Request\Payments\CreatePaymentRequestBuilder
```

**Summary**

Устанавливает IP адрес покупателя.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestBuilder](../classes/YooKassa-Request-Payments-CreatePaymentRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | value  | IPv4 или IPv6-адрес покупателя |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\InvalidPropertyValueTypeException | Выбрасывается если переданный аргумент не является строкой |

**Returns:** \YooKassa\Request\Payments\CreatePaymentRequestBuilder - Инстанс текущего билдера


<a name="method_setConfirmation" class="anchor"></a>
#### public setConfirmation() : \YooKassa\Request\Payments\CreatePaymentRequestBuilder

```php
public setConfirmation(null|\YooKassa\Request\Payments\ConfirmationAttributes\AbstractConfirmationAttributes|array|string $value) : \YooKassa\Request\Payments\CreatePaymentRequestBuilder
```

**Summary**

Устанавливает способ подтверждения платежа.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestBuilder](../classes/YooKassa-Request-Payments-CreatePaymentRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR \YooKassa\Request\Payments\ConfirmationAttributes\AbstractConfirmationAttributes OR array OR string</code> | value  | Способ подтверждения платежа |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\InvalidPropertyValueTypeException | Выбрасывается если переданное значение не является объектом типа AbstractConfirmationAttributes или null |

**Returns:** \YooKassa\Request\Payments\CreatePaymentRequestBuilder - Инстанс текущего билдера


<a name="method_setCurrency" class="anchor"></a>
#### public setCurrency() : self

```php
public setCurrency(string $value) : self
```

**Summary**

Устанавливает валюту в которой будет происходить подтверждение оплаты заказа.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\AbstractPaymentRequestBuilder](../classes/YooKassa-Request-Payments-AbstractPaymentRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | value  | Валюта в которой подтверждается оплата |

**Returns:** self - Инстанс билдера запросов


<a name="method_setDeal" class="anchor"></a>
#### public setDeal() : \YooKassa\Request\Payments\CreatePaymentRequestBuilder

```php
public setDeal(null|array|\YooKassa\Model\Deal\PaymentDealInfo $value) : \YooKassa\Request\Payments\CreatePaymentRequestBuilder
```

**Summary**

Устанавливает сделку.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestBuilder](../classes/YooKassa-Request-Payments-CreatePaymentRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR array OR \YooKassa\Model\Deal\PaymentDealInfo</code> | value  | Данные о сделке, в составе которой проходит платеж |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\InvalidPropertyValueTypeException |  |

**Returns:** \YooKassa\Request\Payments\CreatePaymentRequestBuilder - Инстанс билдера запросов


<a name="method_setDescription" class="anchor"></a>
#### public setDescription() : \YooKassa\Request\Payments\CreatePaymentRequestBuilder

```php
public setDescription(string|null $value) : \YooKassa\Request\Payments\CreatePaymentRequestBuilder
```

**Summary**

Устанавливает описание транзакции.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestBuilder](../classes/YooKassa-Request-Payments-CreatePaymentRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | value  | Описание транзакции |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\InvalidPropertyValueException | Выбрасывается если переданное значение превышает допустимую длину |
| \YooKassa\Common\Exceptions\InvalidPropertyValueTypeException | Выбрасывается если переданное значение не является строкой |

**Returns:** \YooKassa\Request\Payments\CreatePaymentRequestBuilder - Инстанс текущего билдера


<a name="method_setFraudData" class="anchor"></a>
#### (deprecated) - public setFraudData() : \YooKassa\Request\Payments\CreatePaymentRequestBuilder

```php
public setFraudData(null|array|\YooKassa\Request\Payments\FraudData $value) : \YooKassa\Request\Payments\CreatePaymentRequestBuilder
```

**Summary**

Устанавливает информацию для проверки операции на мошенничество.

**Deprecated**
DeprecatedБольше не поддерживается. Вместо него нужно использовать `setReceiver()`
**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestBuilder](../classes/YooKassa-Request-Payments-CreatePaymentRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR array OR \YooKassa\Request\Payments\FraudData</code> | value  | Информация для проверки операции на мошенничество |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\InvalidPropertyValueTypeException |  |

**Returns:** \YooKassa\Request\Payments\CreatePaymentRequestBuilder - Инстанс билдера запросов


<a name="method_setGatewayId" class="anchor"></a>
#### public setGatewayId() : \YooKassa\Request\Payments\CreatePaymentRequestBuilder

```php
public setGatewayId(string $value) : \YooKassa\Request\Payments\CreatePaymentRequestBuilder
```

**Summary**

Устанавливает идентификатор шлюза.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestBuilder](../classes/YooKassa-Request-Payments-CreatePaymentRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | value  | Идентификатор шлюза |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\EmptyPropertyValueException | Выбрасывается если было передано пустое значение |
| \YooKassa\Common\Exceptions\InvalidPropertyValueTypeException | Выбрасывается если было передано не строковое значение |

**Returns:** \YooKassa\Request\Payments\CreatePaymentRequestBuilder - Инстанс текущего билдера


<a name="method_setMerchantCustomerId" class="anchor"></a>
#### public setMerchantCustomerId() : self

```php
public setMerchantCustomerId(string|null $value) : self
```

**Summary**

Устанавливает идентификатор покупателя в вашей системе.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestBuilder](../classes/YooKassa-Request-Payments-CreatePaymentRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | value  | Идентификатор покупателя в вашей системе, например электронная почта или номер телефона. Не более 200 символов |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\InvalidPropertyValueTypeException | Выбрасывается если переданный аргумент не является строкой |

**Returns:** self - 


<a name="method_setMetadata" class="anchor"></a>
#### public setMetadata() : \YooKassa\Request\Payments\CreatePaymentRequestBuilder

```php
public setMetadata(null|array|\YooKassa\Model\Metadata $value) : \YooKassa\Request\Payments\CreatePaymentRequestBuilder
```

**Summary**

Устанавливает метаданные, привязанные к платежу.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestBuilder](../classes/YooKassa-Request-Payments-CreatePaymentRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR array OR \YooKassa\Model\Metadata</code> | value  | Метаданные платежа, устанавливаемые мерчантом |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\InvalidPropertyValueTypeException | Выбрасывается если переданные данные не удалось интерпретировать как метаданные платежа |

**Returns:** \YooKassa\Request\Payments\CreatePaymentRequestBuilder - Инстанс текущего билдера


<a name="method_setOptions" class="anchor"></a>
#### public setOptions() : \YooKassa\Common\AbstractRequestBuilder

```php
public setOptions(iterable|null $options) : \YooKassa\Common\AbstractRequestBuilder
```

**Summary**

Устанавливает свойства запроса из массива.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractRequestBuilder](../classes/YooKassa-Common-AbstractRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">iterable OR null</code> | options  | Массив свойств запроса |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \InvalidArgumentException | Выбрасывается если аргумент не массив и не итерируемый объект |
| \YooKassa\Common\Exceptions\InvalidPropertyException | Выбрасывается если не удалось установить один из параметров, переданных в массиве настроек |

**Returns:** \YooKassa\Common\AbstractRequestBuilder - Инстанс текущего билдера запросов


<a name="method_setPaymentMethodData" class="anchor"></a>
#### public setPaymentMethodData() : \YooKassa\Request\Payments\CreatePaymentRequestBuilder

```php
public setPaymentMethodData(null|\YooKassa\Request\Payments\PaymentData\AbstractPaymentData|array|string $value, array|null $options = null) : \YooKassa\Request\Payments\CreatePaymentRequestBuilder
```

**Summary**

Устанавливает объект с информацией для создания метода оплаты.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestBuilder](../classes/YooKassa-Request-Payments-CreatePaymentRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR \YooKassa\Request\Payments\PaymentData\AbstractPaymentData OR array OR string</code> | value  | Объект создания метода оплаты или null |
| <code lang="php">array OR null</code> | options  | Настройки способа оплаты в виде ассоциативного массива |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\InvalidPropertyValueTypeException | Выбрасывается если был передан объект невалидного типа |

**Returns:** \YooKassa\Request\Payments\CreatePaymentRequestBuilder - Инстанс текущего билдера


<a name="method_setPaymentMethodId" class="anchor"></a>
#### public setPaymentMethodId() : \YooKassa\Request\Payments\CreatePaymentRequestBuilder

```php
public setPaymentMethodId(string|null $value) : \YooKassa\Request\Payments\CreatePaymentRequestBuilder
```

**Summary**

Устанавливает идентификатор записи о сохранённых данных покупателя.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestBuilder](../classes/YooKassa-Request-Payments-CreatePaymentRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | value  | Идентификатор записи о сохраненных платежных данных покупателя |

**Returns:** \YooKassa\Request\Payments\CreatePaymentRequestBuilder - Инстанс текущего билдера


<a name="method_setPaymentOrder" class="anchor"></a>
#### public setPaymentOrder() : \YooKassa\Request\Payments\CreatePaymentRequestBuilder

```php
public setPaymentOrder(null|array|\YooKassa\Request\Payments\PaymentOrderData\AbstractPaymentOrder $value) : \YooKassa\Request\Payments\CreatePaymentRequestBuilder
```

**Summary**

Устанавливает платежное поручение.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestBuilder](../classes/YooKassa-Request-Payments-CreatePaymentRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR array OR \YooKassa\Request\Payments\PaymentOrderData\AbstractPaymentOrder</code> | value  | Платежное поручение |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\InvalidPropertyValueTypeException |  |

**Returns:** \YooKassa\Request\Payments\CreatePaymentRequestBuilder - Инстанс билдера запросов


<a name="method_setPaymentToken" class="anchor"></a>
#### public setPaymentToken() : \YooKassa\Request\Payments\CreatePaymentRequestBuilder

```php
public setPaymentToken(string|null $value) : \YooKassa\Request\Payments\CreatePaymentRequestBuilder
```

**Summary**

Устанавливает одноразовый токен для проведения оплаты.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestBuilder](../classes/YooKassa-Request-Payments-CreatePaymentRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | value  | Одноразовый токен для проведения оплаты |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\InvalidPropertyValueException | Выбрасывается если переданное значение превышает допустимую длину |
| \YooKassa\Common\Exceptions\InvalidPropertyValueTypeException | Выбрасывается если переданное значение не является строкой |

**Returns:** \YooKassa\Request\Payments\CreatePaymentRequestBuilder - Инстанс текущего билдера


<a name="method_setReceipt" class="anchor"></a>
#### public setReceipt() : self

```php
public setReceipt(array|\YooKassa\Model\Receipt\ReceiptInterface $value) : self
```

**Summary**

Устанавливает чек.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\AbstractPaymentRequestBuilder](../classes/YooKassa-Request-Payments-AbstractPaymentRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \YooKassa\Model\Receipt\ReceiptInterface</code> | value  | Инстанс чека или ассоциативный массив с данными чека |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\InvalidPropertyValueTypeException | Генерируется если было передано значение невалидного типа |

**Returns:** self - Инстанс билдера запросов


<a name="method_setReceiptEmail" class="anchor"></a>
#### public setReceiptEmail() : self

```php
public setReceiptEmail(string|null $value) : self
```

**Summary**

Устанавливает адрес электронной почты получателя чека.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\AbstractPaymentRequestBuilder](../classes/YooKassa-Request-Payments-AbstractPaymentRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | value  | Email получателя чека |

**Returns:** self - Инстанс билдера запросов


<a name="method_setReceiptIndustryDetails" class="anchor"></a>
#### public setReceiptIndustryDetails() : self

```php
public setReceiptIndustryDetails(array|\YooKassa\Model\Receipt\IndustryDetails[]|null $value) : self
```

**Summary**

Устанавливает отраслевой реквизит чека.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestBuilder](../classes/YooKassa-Request-Payments-CreatePaymentRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \YooKassa\Model\Receipt\IndustryDetails[] OR null</code> | value  | Отраслевой реквизит чека |

**Returns:** self - Инстанс билдера запросов


<a name="method_setReceiptItems" class="anchor"></a>
#### public setReceiptItems() : self

```php
public setReceiptItems(array $value = []) : self
```

**Summary**

Устанавливает список товаров для создания чека.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\AbstractPaymentRequestBuilder](../classes/YooKassa-Request-Payments-AbstractPaymentRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array</code> | value  | Массив товаров в заказе |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\InvalidPropertyValueException | Выбрасывается если хотя бы один из товаров имеет неверную структуру |

**Returns:** self - Инстанс билдера запросов


<a name="method_setReceiptOperationalDetails" class="anchor"></a>
#### public setReceiptOperationalDetails() : self

```php
public setReceiptOperationalDetails(array|\YooKassa\Model\Receipt\IndustryDetails[]|null $value) : self
```

**Summary**

Устанавливает отраслевой реквизит чека.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestBuilder](../classes/YooKassa-Request-Payments-CreatePaymentRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \YooKassa\Model\Receipt\IndustryDetails[] OR null</code> | value  | Отраслевой реквизит чека |

**Returns:** self - Инстанс билдера запросов


<a name="method_setReceiptPhone" class="anchor"></a>
#### public setReceiptPhone() : self

```php
public setReceiptPhone(string|null $value) : self
```

**Summary**

Устанавливает телефон получателя чека.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\AbstractPaymentRequestBuilder](../classes/YooKassa-Request-Payments-AbstractPaymentRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | value  | Телефон получателя чека |

**Returns:** self - Инстанс билдера запросов


<a name="method_setReceiver" class="anchor"></a>
#### public setReceiver() : \YooKassa\Request\Payments\CreatePaymentRequestBuilder

```php
public setReceiver(null|array|\YooKassa\Request\Payments\ReceiverData\AbstractReceiver $value) : \YooKassa\Request\Payments\CreatePaymentRequestBuilder
```

**Summary**

Устанавливает реквизиты получателя оплаты.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestBuilder](../classes/YooKassa-Request-Payments-CreatePaymentRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR array OR \YooKassa\Request\Payments\ReceiverData\AbstractReceiver</code> | value  | Реквизиты получателя оплаты при пополнении электронного кошелька, банковского счета или баланса телефона |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\InvalidPropertyValueTypeException |  |

**Returns:** \YooKassa\Request\Payments\CreatePaymentRequestBuilder - Инстанс билдера запросов


<a name="method_setRecipient" class="anchor"></a>
#### public setRecipient() : \YooKassa\Request\Payments\CreatePaymentRequestBuilder

```php
public setRecipient(array|\YooKassa\Request\Payments\Recipient|null $value) : \YooKassa\Request\Payments\CreatePaymentRequestBuilder
```

**Summary**

Устанавливает получателя платежа из объекта или ассоциативного массива.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestBuilder](../classes/YooKassa-Request-Payments-CreatePaymentRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \YooKassa\Request\Payments\Recipient OR null</code> | value  | Получатель платежа |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\InvalidPropertyValueTypeException | Выбрасывается если передан аргумент не валидного типа |

**Returns:** \YooKassa\Request\Payments\CreatePaymentRequestBuilder - 


<a name="method_setSavePaymentMethod" class="anchor"></a>
#### public setSavePaymentMethod() : \YooKassa\Request\Payments\CreatePaymentRequestBuilder

```php
public setSavePaymentMethod(bool|null $value = null) : \YooKassa\Request\Payments\CreatePaymentRequestBuilder
```

**Summary**

Устанавливает флаг сохранения платёжных данных. Значение true инициирует создание многоразового payment_method.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestBuilder](../classes/YooKassa-Request-Payments-CreatePaymentRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">bool OR null</code> | value  | Сохранить платежные данные для последующего использования |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\InvalidPropertyValueTypeException | Генерируется если переданный аргумент не кастится в bool |

**Returns:** \YooKassa\Request\Payments\CreatePaymentRequestBuilder - Инстанс текущего билдера


<a name="method_setTaxSystemCode" class="anchor"></a>
#### public setTaxSystemCode() : self

```php
public setTaxSystemCode(int|null $value) : self
```

**Summary**

Устанавливает код системы налогообложения.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\AbstractPaymentRequestBuilder](../classes/YooKassa-Request-Payments-AbstractPaymentRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">int OR null</code> | value  | Код системы налогообложения. Число 1-6. |

**Returns:** self - Инстанс билдера запросов


<a name="method_setTransfers" class="anchor"></a>
#### public setTransfers() : self

```php
public setTransfers(array|\YooKassa\Request\Payments\TransferDataInterface[]|\YooKassa\Common\ListObjectInterface $value) : self
```

**Summary**

Устанавливает трансферы.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\AbstractPaymentRequestBuilder](../classes/YooKassa-Request-Payments-AbstractPaymentRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \YooKassa\Request\Payments\TransferDataInterface[] OR \YooKassa\Common\ListObjectInterface</code> | value  | Массив трансферов |

**Returns:** self - Инстанс билдера запросов


<a name="method_getPaymentDataFactory" class="anchor"></a>
#### protected getPaymentDataFactory() : \YooKassa\Request\Payments\PaymentData\PaymentDataFactory

```php
protected getPaymentDataFactory() : \YooKassa\Request\Payments\PaymentData\PaymentDataFactory
```

**Summary**

Возвращает фабрику методов проведения платежей.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestBuilder](../classes/YooKassa-Request-Payments-CreatePaymentRequestBuilder.md)

**Returns:** \YooKassa\Request\Payments\PaymentData\PaymentDataFactory - Фабрика методов проведения платежей


<a name="method_initCurrentObject" class="anchor"></a>
#### protected initCurrentObject() : \YooKassa\Request\Payments\CreatePaymentRequest

```php
protected initCurrentObject() : \YooKassa\Request\Payments\CreatePaymentRequest
```

**Summary**

Инициализирует объект запроса, который в дальнейшем будет собираться билдером

**Details:**
* Inherited From: [\YooKassa\Request\Payments\CreatePaymentRequestBuilder](../classes/YooKassa-Request-Payments-CreatePaymentRequestBuilder.md)

**Returns:** \YooKassa\Request\Payments\CreatePaymentRequest - Инстанс собираемого объекта запроса к API



---

### Top Namespaces

* [\YooKassa](../namespaces/yookassa.md)

---

### Reports
* [Errors - 0](../reports/errors.md)
* [Markers - 0](../reports/markers.md)
* [Deprecated - 40](../reports/deprecated.md)

---

This document was automatically generated from source code comments on 2025-09-04 using [phpDocumentor](http://www.phpdoc.org/)

&copy; 2025 YooMoney