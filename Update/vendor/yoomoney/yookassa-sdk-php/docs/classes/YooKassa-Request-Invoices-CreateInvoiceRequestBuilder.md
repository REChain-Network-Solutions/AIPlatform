# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Request\Invoices\CreateInvoiceRequestBuilder
### Namespace: [\YooKassa\Request\Invoices](../namespaces/yookassa-request-invoices.md)
---
**Summary:**

Класс, представляющий модель CreateInvoiceRequestBuilder.

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
| protected | [$currentObject](../classes/YooKassa-Request-Invoices-CreateInvoiceRequestBuilder.md#property_currentObject) |  | Собираемый объект запроса. |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [__construct()](../classes/YooKassa-Common-AbstractRequestBuilder.md#method___construct) |  | Конструктор, инициализирует пустой запрос, который в будущем начнём собирать. |
| public | [addCartItem()](../classes/YooKassa-Request-Invoices-CreateInvoiceRequestBuilder.md#method_addCartItem) |  | Добавляет cart_item. |
| public | [build()](../classes/YooKassa-Request-Invoices-CreateInvoiceRequestBuilder.md#method_build) |  | Строит и возвращает объект запроса для отправки в API ЮKassa. |
| public | [setCart()](../classes/YooKassa-Request-Invoices-CreateInvoiceRequestBuilder.md#method_setCart) |  | Устанавливает cart. |
| public | [setDeliveryMethodData()](../classes/YooKassa-Request-Invoices-CreateInvoiceRequestBuilder.md#method_setDeliveryMethodData) |  | Устанавливает delivery_method_data. |
| public | [setDescription()](../classes/YooKassa-Request-Invoices-CreateInvoiceRequestBuilder.md#method_setDescription) |  | Устанавливает описание транзакции. |
| public | [setExpiresAt()](../classes/YooKassa-Request-Invoices-CreateInvoiceRequestBuilder.md#method_setExpiresAt) |  | Устанавливает expires_at. |
| public | [setLocale()](../classes/YooKassa-Request-Invoices-CreateInvoiceRequestBuilder.md#method_setLocale) |  | Устанавливает locale. |
| public | [setMetadata()](../classes/YooKassa-Request-Invoices-CreateInvoiceRequestBuilder.md#method_setMetadata) |  | Устанавливает метаданные, привязанные к счету. |
| public | [setOptions()](../classes/YooKassa-Common-AbstractRequestBuilder.md#method_setOptions) |  | Устанавливает свойства запроса из массива. |
| public | [setPaymentData()](../classes/YooKassa-Request-Invoices-CreateInvoiceRequestBuilder.md#method_setPaymentData) |  | Устанавливает payment_data. |
| protected | [initCurrentObject()](../classes/YooKassa-Request-Invoices-CreateInvoiceRequestBuilder.md#method_initCurrentObject) |  | Инициализирует объект запроса, который в дальнейшем будет собираться билдером |

---
### Details
* File: [lib/Request/Invoices/CreateInvoiceRequestBuilder.php](../../lib/Request/Invoices/CreateInvoiceRequestBuilder.php)
* Package: YooKassa\Request
* Class Hierarchy: 
  * [\YooKassa\Common\AbstractRequestBuilder](../classes/YooKassa-Common-AbstractRequestBuilder.md)
  * \YooKassa\Request\Invoices\CreateInvoiceRequestBuilder

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
<a name="property_currentObject"></a>
#### protected $currentObject : ?\YooKassa\Common\AbstractRequestInterface
---
**Summary**

Собираемый объект запроса.

**Type:** <a href="../?\YooKassa\Common\AbstractRequestInterface"><abbr title="?\YooKassa\Common\AbstractRequestInterface">AbstractRequestInterface</abbr></a>

**Details:**



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


<a name="method_addCartItem" class="anchor"></a>
#### public addCartItem() : self

```php
public addCartItem(\YooKassa\Model\Invoice\LineItem|array|null $value = null) : self
```

**Summary**

Добавляет cart_item.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\CreateInvoiceRequestBuilder](../classes/YooKassa-Request-Invoices-CreateInvoiceRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\Invoice\LineItem OR array OR null</code> | value  | Товар или услуга. |

**Returns:** self - 


<a name="method_build" class="anchor"></a>
#### public build() : \YooKassa\Request\Invoices\CreateInvoiceRequestInterface|\YooKassa\Common\AbstractRequestInterface

```php
public build(null|array $options = null) : \YooKassa\Request\Invoices\CreateInvoiceRequestInterface|\YooKassa\Common\AbstractRequestInterface
```

**Summary**

Строит и возвращает объект запроса для отправки в API ЮKassa.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\CreateInvoiceRequestBuilder](../classes/YooKassa-Request-Invoices-CreateInvoiceRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR array</code> | options  | Массив параметров для установки в объект запроса |

**Returns:** \YooKassa\Request\Invoices\CreateInvoiceRequestInterface|\YooKassa\Common\AbstractRequestInterface - Инстанс объекта запроса


<a name="method_setCart" class="anchor"></a>
#### public setCart() : self

```php
public setCart(\YooKassa\Common\ListObjectInterface|array|null $value = null) : self
```

**Summary**

Устанавливает cart.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\CreateInvoiceRequestBuilder](../classes/YooKassa-Request-Invoices-CreateInvoiceRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Common\ListObjectInterface OR array OR null</code> | value  | Корзина заказа — список товаров или услуг, который отобразится на странице счета перед оплатой. |

**Returns:** self - 


<a name="method_setDeliveryMethodData" class="anchor"></a>
#### public setDeliveryMethodData() : self

```php
public setDeliveryMethodData(\YooKassa\Request\Invoices\DeliveryMethodData\AbstractDeliveryMethodData|array|null $value = null) : self
```

**Summary**

Устанавливает delivery_method_data.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\CreateInvoiceRequestBuilder](../classes/YooKassa-Request-Invoices-CreateInvoiceRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Request\Invoices\DeliveryMethodData\AbstractDeliveryMethodData OR array OR null</code> | value  | Данные о способе доставки счета пользователю |

**Returns:** self - 


<a name="method_setDescription" class="anchor"></a>
#### public setDescription() : \YooKassa\Request\Invoices\CreateInvoiceRequestBuilder

```php
public setDescription(string|null $value) : \YooKassa\Request\Invoices\CreateInvoiceRequestBuilder
```

**Summary**

Устанавливает описание транзакции.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\CreateInvoiceRequestBuilder](../classes/YooKassa-Request-Invoices-CreateInvoiceRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | value  | Описание транзакции |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\InvalidPropertyValueException | Выбрасывается если переданное значение превышает допустимую длину |
| \YooKassa\Common\Exceptions\InvalidPropertyValueTypeException | Выбрасывается если переданное значение не является строкой |

**Returns:** \YooKassa\Request\Invoices\CreateInvoiceRequestBuilder - Инстанс текущего билдера


<a name="method_setExpiresAt" class="anchor"></a>
#### public setExpiresAt() : self

```php
public setExpiresAt(\DateTime|string|null $value = null) : self
```

**Summary**

Устанавливает expires_at.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\CreateInvoiceRequestBuilder](../classes/YooKassa-Request-Invoices-CreateInvoiceRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\DateTime OR string OR null</code> | value  | Срок действия счета — дата и время, до которых можно оплатить выставленный счет. Указывается по [UTC](https://ru.wikipedia.org/wiki/Всемирное_координированное_время) и передается в формате [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601). Пример: ~`2024-10-18T10:51:18.139Z` |

**Returns:** self - 


<a name="method_setLocale" class="anchor"></a>
#### public setLocale() : self

```php
public setLocale(string|null $value = null) : self
```

**Summary**

Устанавливает locale.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\CreateInvoiceRequestBuilder](../classes/YooKassa-Request-Invoices-CreateInvoiceRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | value  | Язык интерфейса, писем и смс, которые будет видеть или получать пользователь |

**Returns:** self - 


<a name="method_setMetadata" class="anchor"></a>
#### public setMetadata() : \YooKassa\Request\Invoices\CreateInvoiceRequestBuilder

```php
public setMetadata(null|array|\YooKassa\Model\Metadata $value) : \YooKassa\Request\Invoices\CreateInvoiceRequestBuilder
```

**Summary**

Устанавливает метаданные, привязанные к счету.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\CreateInvoiceRequestBuilder](../classes/YooKassa-Request-Invoices-CreateInvoiceRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR array OR \YooKassa\Model\Metadata</code> | value  | Метаданные платежа, устанавливаемые мерчантом |

##### Throws:
| Type | Description |
| ---- | ----------- |
| \YooKassa\Common\Exceptions\InvalidPropertyValueTypeException | Выбрасывается если переданные данные не удалось интерпретировать как метаданные платежа |

**Returns:** \YooKassa\Request\Invoices\CreateInvoiceRequestBuilder - Инстанс текущего билдера


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


<a name="method_setPaymentData" class="anchor"></a>
#### public setPaymentData() : self

```php
public setPaymentData(\YooKassa\Request\Invoices\PaymentData|array|null $value = null) : self
```

**Summary**

Устанавливает payment_data.

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\CreateInvoiceRequestBuilder](../classes/YooKassa-Request-Invoices-CreateInvoiceRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Request\Invoices\PaymentData OR array OR null</code> | value  | Данные для проведения платежа по выставленному счету |

**Returns:** self - 


<a name="method_initCurrentObject" class="anchor"></a>
#### protected initCurrentObject() : \YooKassa\Request\Invoices\CreateInvoiceRequest

```php
protected initCurrentObject() : \YooKassa\Request\Invoices\CreateInvoiceRequest
```

**Summary**

Инициализирует объект запроса, который в дальнейшем будет собираться билдером

**Details:**
* Inherited From: [\YooKassa\Request\Invoices\CreateInvoiceRequestBuilder](../classes/YooKassa-Request-Invoices-CreateInvoiceRequestBuilder.md)

**Returns:** \YooKassa\Request\Invoices\CreateInvoiceRequest - Инстанс собираемого объекта запроса к API



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