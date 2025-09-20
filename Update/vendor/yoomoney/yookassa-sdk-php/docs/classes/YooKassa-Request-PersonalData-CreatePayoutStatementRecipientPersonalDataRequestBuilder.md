# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Request\PersonalData\CreatePayoutStatementRecipientPersonalDataRequestBuilder
### Namespace: [\YooKassa\Request\PersonalData](../namespaces/yookassa-request-personaldata.md)
---
**Summary:**

Класс, представляющий модель CreatePayoutStatementRecipientPersonalDataRequestBuilder.

**Description:**

Класс билдера объектов запросов к API на создание платежа.

---
### Examples
Пример использования билдера

```php
try {
    $personalDataBuilder = \YooKassa\Request\PersonalData\PersonalDataType\SbpPayoutRecipientPersonalDataRequest::builder();
    $personalDataBuilder
        ->setFirstName('Иван')
        ->setLastName('Иванов')
        ->setMiddleName('Иванович')
        ->setMetadata(['recipient_id' => '37'])
    ;

    // Создаем объект запроса
    $request = $personalDataBuilder->build();

    $idempotenceKey = uniqid('', true);
    $response = $client->createPersonalData($request, $idempotenceKey);
} catch (Exception $e) {
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
| protected | [$currentObject](../classes/YooKassa-Request-PersonalData-CreatePayoutStatementRecipientPersonalDataRequestBuilder.md#property_currentObject) |  | Собираемый объект запроса. |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [__construct()](../classes/YooKassa-Common-AbstractRequestBuilder.md#method___construct) |  | Конструктор, инициализирует пустой запрос, который в будущем начнём собирать. |
| public | [build()](../classes/YooKassa-Request-PersonalData-CreatePayoutStatementRecipientPersonalDataRequestBuilder.md#method_build) |  | Строит и возвращает объект запроса для отправки в API ЮKassa. |
| public | [setBirthdate()](../classes/YooKassa-Request-PersonalData-CreatePayoutStatementRecipientPersonalDataRequestBuilder.md#method_setBirthdate) |  | Устанавливает дату рождения. |
| public | [setFirstName()](../classes/YooKassa-Request-PersonalData-CreatePayoutStatementRecipientPersonalDataRequestBuilder.md#method_setFirstName) |  | Устанавливает имя пользователя. |
| public | [setLastName()](../classes/YooKassa-Request-PersonalData-CreatePayoutStatementRecipientPersonalDataRequestBuilder.md#method_setLastName) |  | Устанавливает фамилию пользователя. |
| public | [setMetadata()](../classes/YooKassa-Request-PersonalData-CreatePayoutStatementRecipientPersonalDataRequestBuilder.md#method_setMetadata) |  | Устанавливает метаданные, привязанные к платежу. |
| public | [setMiddleName()](../classes/YooKassa-Request-PersonalData-CreatePayoutStatementRecipientPersonalDataRequestBuilder.md#method_setMiddleName) |  | Устанавливает отчество пользователя. |
| public | [setOptions()](../classes/YooKassa-Common-AbstractRequestBuilder.md#method_setOptions) |  | Устанавливает свойства запроса из массива. |
| protected | [initCurrentObject()](../classes/YooKassa-Request-PersonalData-CreatePayoutStatementRecipientPersonalDataRequestBuilder.md#method_initCurrentObject) |  | Инициализирует объект запроса, который в дальнейшем будет собираться билдером |

---
### Details
* File: [lib/Request/PersonalData/CreatePayoutStatementRecipientPersonalDataRequestBuilder.php](../../lib/Request/PersonalData/CreatePayoutStatementRecipientPersonalDataRequestBuilder.php)
* Package: YooKassa\Request
* Class Hierarchy: 
  * [\YooKassa\Common\AbstractRequestBuilder](../classes/YooKassa-Common-AbstractRequestBuilder.md)
  * \YooKassa\Request\PersonalData\CreatePayoutStatementRecipientPersonalDataRequestBuilder

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


<a name="method_build" class="anchor"></a>
#### public build() : \YooKassa\Common\AbstractRequestInterface|\YooKassa\Request\PersonalData\PersonalDataType\PayoutStatementRecipientPersonalDataRequest

```php
public build(null|array $options = null) : \YooKassa\Common\AbstractRequestInterface|\YooKassa\Request\PersonalData\PersonalDataType\PayoutStatementRecipientPersonalDataRequest
```

**Summary**

Строит и возвращает объект запроса для отправки в API ЮKassa.

**Details:**
* Inherited From: [\YooKassa\Request\PersonalData\CreatePayoutStatementRecipientPersonalDataRequestBuilder](../classes/YooKassa-Request-PersonalData-CreatePayoutStatementRecipientPersonalDataRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR array</code> | options  | Массив параметров для установки в объект запроса |

**Returns:** \YooKassa\Common\AbstractRequestInterface|\YooKassa\Request\PersonalData\PersonalDataType\PayoutStatementRecipientPersonalDataRequest - Инстанс объекта запроса


<a name="method_setBirthdate" class="anchor"></a>
#### public setBirthdate() : self

```php
public setBirthdate(\DateTime|string|null $value = null) : self
```

**Summary**

Устанавливает дату рождения.

**Details:**
* Inherited From: [\YooKassa\Request\PersonalData\CreatePayoutStatementRecipientPersonalDataRequestBuilder](../classes/YooKassa-Request-PersonalData-CreatePayoutStatementRecipientPersonalDataRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\DateTime OR string OR null</code> | value  | Дата рождения. Передается в формате [ISO 8601](https://en.wikipedia.org/wiki/ISO_8601) |

**Returns:** self - Инстанс текущего билдера


<a name="method_setFirstName" class="anchor"></a>
#### public setFirstName() : self

```php
public setFirstName(string $value) : self
```

**Summary**

Устанавливает имя пользователя.

**Details:**
* Inherited From: [\YooKassa\Request\PersonalData\CreatePayoutStatementRecipientPersonalDataRequestBuilder](../classes/YooKassa-Request-PersonalData-CreatePayoutStatementRecipientPersonalDataRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | value  | имя пользователя |

**Returns:** self - 


<a name="method_setLastName" class="anchor"></a>
#### public setLastName() : self

```php
public setLastName(string $value) : self
```

**Summary**

Устанавливает фамилию пользователя.

**Details:**
* Inherited From: [\YooKassa\Request\PersonalData\CreatePayoutStatementRecipientPersonalDataRequestBuilder](../classes/YooKassa-Request-PersonalData-CreatePayoutStatementRecipientPersonalDataRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | value  | фамилия пользователя |

**Returns:** self - 


<a name="method_setMetadata" class="anchor"></a>
#### public setMetadata() : self

```php
public setMetadata(null|array|\YooKassa\Model\Metadata $value) : self
```

**Summary**

Устанавливает метаданные, привязанные к платежу.

**Details:**
* Inherited From: [\YooKassa\Request\PersonalData\CreatePayoutStatementRecipientPersonalDataRequestBuilder](../classes/YooKassa-Request-PersonalData-CreatePayoutStatementRecipientPersonalDataRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR array OR \YooKassa\Model\Metadata</code> | value  | Метаданные платежа, устанавливаемые мерчантом |

**Returns:** self - Инстанс текущего билдера


<a name="method_setMiddleName" class="anchor"></a>
#### public setMiddleName() : self

```php
public setMiddleName(null|string $value) : self
```

**Summary**

Устанавливает отчество пользователя.

**Details:**
* Inherited From: [\YooKassa\Request\PersonalData\CreatePayoutStatementRecipientPersonalDataRequestBuilder](../classes/YooKassa-Request-PersonalData-CreatePayoutStatementRecipientPersonalDataRequestBuilder.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">null OR string</code> | value  | Отчество пользователя |

**Returns:** self - 


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


<a name="method_initCurrentObject" class="anchor"></a>
#### protected initCurrentObject() : \YooKassa\Request\PersonalData\PersonalDataType\PayoutStatementRecipientPersonalDataRequest

```php
protected initCurrentObject() : \YooKassa\Request\PersonalData\PersonalDataType\PayoutStatementRecipientPersonalDataRequest
```

**Summary**

Инициализирует объект запроса, который в дальнейшем будет собираться билдером

**Details:**
* Inherited From: [\YooKassa\Request\PersonalData\CreatePayoutStatementRecipientPersonalDataRequestBuilder](../classes/YooKassa-Request-PersonalData-CreatePayoutStatementRecipientPersonalDataRequestBuilder.md)

**Returns:** \YooKassa\Request\PersonalData\PersonalDataType\PayoutStatementRecipientPersonalDataRequest - Инстанс собираемого объекта запроса к API



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