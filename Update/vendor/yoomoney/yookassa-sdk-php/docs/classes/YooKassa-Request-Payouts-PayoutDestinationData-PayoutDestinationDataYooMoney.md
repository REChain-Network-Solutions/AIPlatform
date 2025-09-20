# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Request\Payouts\PayoutDestinationData\PayoutDestinationDataYooMoney
### Namespace: [\YooKassa\Request\Payouts\PayoutDestinationData](../namespaces/yookassa-request-payouts-payoutdestinationdata.md)
---
**Summary:**

Класс, представляющий модель PayoutDestinationDataYooMoney.

**Description:**

Метод оплаты, при оплате через ЮMoney

---
### Constants
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [MAX_LENGTH_ACCOUNT_NUMBER](../classes/YooKassa-Request-Payouts-PayoutDestinationData-PayoutDestinationDataYooMoney.md#constant_MAX_LENGTH_ACCOUNT_NUMBER) |  |  |
| public | [MIN_LENGTH_ACCOUNT_NUMBER](../classes/YooKassa-Request-Payouts-PayoutDestinationData-PayoutDestinationDataYooMoney.md#constant_MIN_LENGTH_ACCOUNT_NUMBER) |  |  |

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [$account_number](../classes/YooKassa-Request-Payouts-PayoutDestinationData-PayoutDestinationDataYooMoney.md#property_account_number) |  | Номер кошелька в ЮMoney, с которого была произведена оплата |
| public | [$accountNumber](../classes/YooKassa-Request-Payouts-PayoutDestinationData-PayoutDestinationDataYooMoney.md#property_accountNumber) |  | Номер кошелька в ЮMoney, с которого была произведена оплата |
| public | [$type](../classes/YooKassa-Request-Payouts-PayoutDestinationData-PayoutDestinationDataYooMoney.md#property_type) |  | Тип объекта |
| public | [$type](../classes/YooKassa-Request-Payouts-PayoutDestinationData-AbstractPayoutDestinationData.md#property_type) |  | Тип метода оплаты |
| public | [$type](../classes/YooKassa-Model-Payout-AbstractPayoutDestination.md#property_type) |  | Тип метода оплаты |
| protected | [$_type](../classes/YooKassa-Model-Payout-AbstractPayoutDestination.md#property__type) |  | Тип метода оплаты |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [__construct()](../classes/YooKassa-Request-Payouts-PayoutDestinationData-PayoutDestinationDataYooMoney.md#method___construct) |  | Конструктор PayoutDestinationDataYooMoney. |
| public | [__get()](../classes/YooKassa-Common-AbstractObject.md#method___get) |  | Возвращает значение свойства. |
| public | [__isset()](../classes/YooKassa-Common-AbstractObject.md#method___isset) |  | Проверяет наличие свойства. |
| public | [__set()](../classes/YooKassa-Common-AbstractObject.md#method___set) |  | Устанавливает значение свойства. |
| public | [__unset()](../classes/YooKassa-Common-AbstractObject.md#method___unset) |  | Удаляет свойство. |
| public | [fromArray()](../classes/YooKassa-Common-AbstractObject.md#method_fromArray) |  | Устанавливает значения свойств текущего объекта из массива. |
| public | [getAccountNumber()](../classes/YooKassa-Request-Payouts-PayoutDestinationData-PayoutDestinationDataYooMoney.md#method_getAccountNumber) |  | Возвращает номер кошелька в ЮMoney, с которого была произведена оплата. |
| public | [getType()](../classes/YooKassa-Model-Payout-AbstractPayoutDestination.md#method_getType) |  | Возвращает тип метода оплаты. |
| public | [getValidator()](../classes/YooKassa-Common-AbstractObject.md#method_getValidator) |  |  |
| public | [jsonSerialize()](../classes/YooKassa-Common-AbstractObject.md#method_jsonSerialize) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации. |
| public | [offsetExists()](../classes/YooKassa-Common-AbstractObject.md#method_offsetExists) |  | Проверяет наличие свойства. |
| public | [offsetGet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetGet) |  | Возвращает значение свойства. |
| public | [offsetSet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetSet) |  | Устанавливает значение свойства. |
| public | [offsetUnset()](../classes/YooKassa-Common-AbstractObject.md#method_offsetUnset) |  | Удаляет свойство. |
| public | [setAccountNumber()](../classes/YooKassa-Request-Payouts-PayoutDestinationData-PayoutDestinationDataYooMoney.md#method_setAccountNumber) |  | Устанавливает номер кошелька в ЮMoney, с которого была произведена оплата. |
| public | [setType()](../classes/YooKassa-Model-Payout-AbstractPayoutDestination.md#method_setType) |  | Устанавливает тип метода оплаты. |
| public | [toArray()](../classes/YooKassa-Common-AbstractObject.md#method_toArray) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации Является алиасом метода AbstractObject::jsonSerialize(). |
| protected | [getUnknownProperties()](../classes/YooKassa-Common-AbstractObject.md#method_getUnknownProperties) |  | Возвращает массив свойств которые не существуют, но были заданы у объекта. |
| protected | [validatePropertyValue()](../classes/YooKassa-Common-AbstractObject.md#method_validatePropertyValue) |  |  |

---
### Details
* File: [lib/Request/Payouts/PayoutDestinationData/PayoutDestinationDataYooMoney.php](../../lib/Request/Payouts/PayoutDestinationData/PayoutDestinationDataYooMoney.php)
* Package: YooKassa\Request
* Class Hierarchy:   
  * [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)
  * [\YooKassa\Model\Payout\AbstractPayoutDestination](../classes/YooKassa-Model-Payout-AbstractPayoutDestination.md)
  * [\YooKassa\Request\Payouts\PayoutDestinationData\AbstractPayoutDestinationData](../classes/YooKassa-Request-Payouts-PayoutDestinationData-AbstractPayoutDestinationData.md)
  * \YooKassa\Request\Payouts\PayoutDestinationData\PayoutDestinationDataYooMoney

* See Also:
  * [](https://yookassa.ru/developers/api)

---
### Tags
| Tag | Version | Description |
| --- | ------- | ----------- |
| category |  | Class |
| author |  | cms@yoomoney.ru |

---
## Constants
<a name="constant_MAX_LENGTH_ACCOUNT_NUMBER" class="anchor"></a>
###### MAX_LENGTH_ACCOUNT_NUMBER
```php
MAX_LENGTH_ACCOUNT_NUMBER = 33 : int
```


<a name="constant_MIN_LENGTH_ACCOUNT_NUMBER" class="anchor"></a>
###### MIN_LENGTH_ACCOUNT_NUMBER
```php
MIN_LENGTH_ACCOUNT_NUMBER = 11 : int
```



---
## Properties
<a name="property_account_number"></a>
#### public $account_number : string
---
***Description***

Номер кошелька в ЮMoney, с которого была произведена оплата

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_accountNumber"></a>
#### public $accountNumber : string
---
***Description***

Номер кошелька в ЮMoney, с которого была произведена оплата

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_type"></a>
#### public $type : string
---
***Description***

Тип объекта

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_type"></a>
#### public $type : string
---
***Description***

Тип метода оплаты

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\PayoutDestinationData\AbstractPayoutDestinationData](../classes/YooKassa-Request-Payouts-PayoutDestinationData-AbstractPayoutDestinationData.md)


<a name="property_type"></a>
#### public $type : string
---
***Description***

Тип метода оплаты

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Model\Payout\AbstractPayoutDestination](../classes/YooKassa-Model-Payout-AbstractPayoutDestination.md)


<a name="property__type"></a>
#### protected $_type : ?string
---
**Summary**

Тип метода оплаты

**Type:** <a href="../?string"><abbr title="?string">?string</abbr></a>

**Details:**
* Inherited From: [\YooKassa\Model\Payout\AbstractPayoutDestination](../classes/YooKassa-Model-Payout-AbstractPayoutDestination.md)



---
## Methods
<a name="method___construct" class="anchor"></a>
#### public __construct() : mixed

```php
public __construct(array|null $data = []) : mixed
```

**Summary**

Конструктор PayoutDestinationDataYooMoney.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\PayoutDestinationData\PayoutDestinationDataYooMoney](../classes/YooKassa-Request-Payouts-PayoutDestinationData-PayoutDestinationDataYooMoney.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR null</code> | data  |  |

**Returns:** mixed - 


<a name="method___get" class="anchor"></a>
#### public __get() : mixed

```php
public __get(string $propertyName) : mixed
```

**Summary**

Возвращает значение свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | propertyName  | Имя свойства |

**Returns:** mixed - Значение свойства


<a name="method___isset" class="anchor"></a>
#### public __isset() : bool

```php
public __isset(string $propertyName) : bool
```

**Summary**

Проверяет наличие свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | propertyName  | Имя проверяемого свойства |

**Returns:** bool - True если свойство имеется, false если нет


<a name="method___set" class="anchor"></a>
#### public __set() : void

```php
public __set(string $propertyName, mixed $value) : void
```

**Summary**

Устанавливает значение свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | propertyName  | Имя свойства |
| <code lang="php">mixed</code> | value  | Значение свойства |

**Returns:** void - 


<a name="method___unset" class="anchor"></a>
#### public __unset() : void

```php
public __unset(string $propertyName) : void
```

**Summary**

Удаляет свойство.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | propertyName  | Имя удаляемого свойства |

**Returns:** void - 


<a name="method_fromArray" class="anchor"></a>
#### public fromArray() : void

```php
public fromArray(array|\Traversable $sourceArray) : void
```

**Summary**

Устанавливает значения свойств текущего объекта из массива.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">array OR \Traversable</code> | sourceArray  | Ассоциативный массив с настройками |

**Returns:** void - 


<a name="method_getAccountNumber" class="anchor"></a>
#### public getAccountNumber() : string|null

```php
public getAccountNumber() : string|null
```

**Summary**

Возвращает номер кошелька в ЮMoney, с которого была произведена оплата.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\PayoutDestinationData\PayoutDestinationDataYooMoney](../classes/YooKassa-Request-Payouts-PayoutDestinationData-PayoutDestinationDataYooMoney.md)

**Returns:** string|null - Номер кошелька в ЮMoney


<a name="method_getType" class="anchor"></a>
#### public getType() : string|null

```php
public getType() : string|null
```

**Summary**

Возвращает тип метода оплаты.

**Details:**
* Inherited From: [\YooKassa\Model\Payout\AbstractPayoutDestination](../classes/YooKassa-Model-Payout-AbstractPayoutDestination.md)

**Returns:** string|null - Тип метода оплаты


<a name="method_getValidator" class="anchor"></a>
#### public getValidator() : \YooKassa\Validator\Validator

```php
public getValidator() : \YooKassa\Validator\Validator
```

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** \YooKassa\Validator\Validator - 


<a name="method_jsonSerialize" class="anchor"></a>
#### public jsonSerialize() : array

```php
public jsonSerialize() : array
```

**Summary**

Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** array - Ассоциативный массив со свойствами текущего объекта


<a name="method_offsetExists" class="anchor"></a>
#### public offsetExists() : bool

```php
public offsetExists(string $offset) : bool
```

**Summary**

Проверяет наличие свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | offset  | Имя проверяемого свойства |

**Returns:** bool - True если свойство имеется, false если нет


<a name="method_offsetGet" class="anchor"></a>
#### public offsetGet() : mixed

```php
public offsetGet(string $offset) : mixed
```

**Summary**

Возвращает значение свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | offset  | Имя свойства |

**Returns:** mixed - Значение свойства


<a name="method_offsetSet" class="anchor"></a>
#### public offsetSet() : void

```php
public offsetSet(string $offset, mixed $value) : void
```

**Summary**

Устанавливает значение свойства.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | offset  | Имя свойства |
| <code lang="php">mixed</code> | value  | Значение свойства |

**Returns:** void - 


<a name="method_offsetUnset" class="anchor"></a>
#### public offsetUnset() : void

```php
public offsetUnset(string $offset) : void
```

**Summary**

Удаляет свойство.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | offset  | Имя удаляемого свойства |

**Returns:** void - 


<a name="method_setAccountNumber" class="anchor"></a>
#### public setAccountNumber() : self

```php
public setAccountNumber(string|null $account_number = null) : self
```

**Summary**

Устанавливает номер кошелька в ЮMoney, с которого была произведена оплата.

**Details:**
* Inherited From: [\YooKassa\Request\Payouts\PayoutDestinationData\PayoutDestinationDataYooMoney](../classes/YooKassa-Request-Payouts-PayoutDestinationData-PayoutDestinationDataYooMoney.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | account_number  | Номер кошелька ЮMoney |

**Returns:** self - 


<a name="method_setType" class="anchor"></a>
#### public setType() : self

```php
public setType(string|null $type = null) : self
```

**Summary**

Устанавливает тип метода оплаты.

**Details:**
* Inherited From: [\YooKassa\Model\Payout\AbstractPayoutDestination](../classes/YooKassa-Model-Payout-AbstractPayoutDestination.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | type  | Тип метода оплаты |

**Returns:** self - 


<a name="method_toArray" class="anchor"></a>
#### public toArray() : array

```php
public toArray() : array
```

**Summary**

Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации
Является алиасом метода AbstractObject::jsonSerialize().

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** array - Ассоциативный массив со свойствами текущего объекта


<a name="method_getUnknownProperties" class="anchor"></a>
#### protected getUnknownProperties() : array

```php
protected getUnknownProperties() : array
```

**Summary**

Возвращает массив свойств которые не существуют, но были заданы у объекта.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** array - Ассоциативный массив с не существующими у текущего объекта свойствами


<a name="method_validatePropertyValue" class="anchor"></a>
#### protected validatePropertyValue() : mixed

```php
protected validatePropertyValue(string $propertyName, mixed $propertyValue) : mixed
```

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string</code> | propertyName  |  |
| <code lang="php">mixed</code> | propertyValue  |  |

**Returns:** mixed - 



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