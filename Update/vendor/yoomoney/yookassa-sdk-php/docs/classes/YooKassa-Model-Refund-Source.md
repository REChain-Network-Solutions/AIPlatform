# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Model\Refund\Source
### Namespace: [\YooKassa\Model\Refund](../namespaces/yookassa-model-refund.md)
---
**Summary:**

Класс, представляющий модель RefundSourcesData.

**Description:**

Данные о том, с какого магазина и какую сумму нужно удержать для проведения возврата.
Сейчас в этом параметре можно передать данные только одного магазина.

---
### Constants
* No constants found

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [$account_id](../classes/YooKassa-Model-Refund-Source.md#property_account_id) |  | Идентификатор магазина, для которого вы хотите провести возврат |
| public | [$accountId](../classes/YooKassa-Model-Refund-Source.md#property_accountId) |  | Идентификатор магазина, для которого вы хотите провести возврат |
| public | [$amount](../classes/YooKassa-Model-Refund-Source.md#property_amount) |  | Сумма возврата |
| public | [$platform_fee_amount](../classes/YooKassa-Model-Refund-Source.md#property_platform_fee_amount) |  | Комиссия, которую вы удержали при оплате, и хотите вернуть |
| public | [$platformFeeAmount](../classes/YooKassa-Model-Refund-Source.md#property_platformFeeAmount) |  | Комиссия, которую вы удержали при оплате, и хотите вернуть |

---
### Methods
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [__construct()](../classes/YooKassa-Common-AbstractObject.md#method___construct) |  | AbstractObject constructor. |
| public | [__get()](../classes/YooKassa-Common-AbstractObject.md#method___get) |  | Возвращает значение свойства. |
| public | [__isset()](../classes/YooKassa-Common-AbstractObject.md#method___isset) |  | Проверяет наличие свойства. |
| public | [__set()](../classes/YooKassa-Common-AbstractObject.md#method___set) |  | Устанавливает значение свойства. |
| public | [__unset()](../classes/YooKassa-Common-AbstractObject.md#method___unset) |  | Удаляет свойство. |
| public | [fromArray()](../classes/YooKassa-Common-AbstractObject.md#method_fromArray) |  | Устанавливает значения свойств текущего объекта из массива. |
| public | [getAccountId()](../classes/YooKassa-Model-Refund-Source.md#method_getAccountId) |  | Возвращает id магазина с которого будут списаны средства. |
| public | [getAmount()](../classes/YooKassa-Model-Refund-Source.md#method_getAmount) |  | Возвращает сумму оплаты. |
| public | [getPlatformFeeAmount()](../classes/YooKassa-Model-Refund-Source.md#method_getPlatformFeeAmount) |  | Возвращает комиссию за проданные товары и услуги, которая удерживается с магазина в вашу пользу. |
| public | [getValidator()](../classes/YooKassa-Common-AbstractObject.md#method_getValidator) |  |  |
| public | [hasAmount()](../classes/YooKassa-Model-Refund-Source.md#method_hasAmount) |  | Проверяет, была ли установлена сумма оплаты. |
| public | [hasPlatformFeeAmount()](../classes/YooKassa-Model-Refund-Source.md#method_hasPlatformFeeAmount) |  | Проверяет, была ли установлена комиссия за проданные товары и услуги, которая удерживается с магазина в вашу пользу. |
| public | [jsonSerialize()](../classes/YooKassa-Common-AbstractObject.md#method_jsonSerialize) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации. |
| public | [offsetExists()](../classes/YooKassa-Common-AbstractObject.md#method_offsetExists) |  | Проверяет наличие свойства. |
| public | [offsetGet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetGet) |  | Возвращает значение свойства. |
| public | [offsetSet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetSet) |  | Устанавливает значение свойства. |
| public | [offsetUnset()](../classes/YooKassa-Common-AbstractObject.md#method_offsetUnset) |  | Удаляет свойство. |
| public | [setAccountId()](../classes/YooKassa-Model-Refund-Source.md#method_setAccountId) |  | Устанавливает id магазина-получателя средств. |
| public | [setAmount()](../classes/YooKassa-Model-Refund-Source.md#method_setAmount) |  | Устанавливает сумму оплаты. |
| public | [setPlatformFeeAmount()](../classes/YooKassa-Model-Refund-Source.md#method_setPlatformFeeAmount) |  | Устанавливает комиссию за проданные товары и услуги, которая удерживается с магазина в вашу пользу. |
| public | [toArray()](../classes/YooKassa-Common-AbstractObject.md#method_toArray) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации Является алиасом метода AbstractObject::jsonSerialize(). |
| protected | [getUnknownProperties()](../classes/YooKassa-Common-AbstractObject.md#method_getUnknownProperties) |  | Возвращает массив свойств которые не существуют, но были заданы у объекта. |
| protected | [validatePropertyValue()](../classes/YooKassa-Common-AbstractObject.md#method_validatePropertyValue) |  |  |

---
### Details
* File: [lib/Model/Refund/Source.php](../../lib/Model/Refund/Source.php)
* Package: YooKassa\Model
* Class Hierarchy: 
  * [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)
  * \YooKassa\Model\Refund\Source
* Implements:
  * [\YooKassa\Model\Refund\SourceInterface](../classes/YooKassa-Model-Refund-SourceInterface.md)

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
<a name="property_account_id"></a>
#### public $account_id : string
---
***Description***

Идентификатор магазина, для которого вы хотите провести возврат

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_accountId"></a>
#### public $accountId : string
---
***Description***

Идентификатор магазина, для которого вы хотите провести возврат

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_amount"></a>
#### public $amount : \YooKassa\Model\AmountInterface
---
***Description***

Сумма возврата

**Type:** <a href="../classes/YooKassa-Model-AmountInterface.html"><abbr title="\YooKassa\Model\AmountInterface">AmountInterface</abbr></a>

**Details:**


<a name="property_platform_fee_amount"></a>
#### public $platform_fee_amount : \YooKassa\Model\AmountInterface
---
***Description***

Комиссия, которую вы удержали при оплате, и хотите вернуть

**Type:** <a href="../classes/YooKassa-Model-AmountInterface.html"><abbr title="\YooKassa\Model\AmountInterface">AmountInterface</abbr></a>

**Details:**


<a name="property_platformFeeAmount"></a>
#### public $platformFeeAmount : \YooKassa\Model\AmountInterface
---
***Description***

Комиссия, которую вы удержали при оплате, и хотите вернуть

**Type:** <a href="../classes/YooKassa-Model-AmountInterface.html"><abbr title="\YooKassa\Model\AmountInterface">AmountInterface</abbr></a>

**Details:**



---
## Methods
<a name="method___construct" class="anchor"></a>
#### public __construct() : mixed

```php
public __construct(array|null $data = []) : mixed
```

**Summary**

AbstractObject constructor.

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

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


<a name="method_getAccountId" class="anchor"></a>
#### public getAccountId() : string|null

```php
public getAccountId() : string|null
```

**Summary**

Возвращает id магазина с которого будут списаны средства.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\Source](../classes/YooKassa-Model-Refund-Source.md)

**Returns:** string|null - Id магазина с которого будут списаны средства


<a name="method_getAmount" class="anchor"></a>
#### public getAmount() : \YooKassa\Model\AmountInterface|null

```php
public getAmount() : \YooKassa\Model\AmountInterface|null
```

**Summary**

Возвращает сумму оплаты.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\Source](../classes/YooKassa-Model-Refund-Source.md)

**Returns:** \YooKassa\Model\AmountInterface|null - Сумма оплаты


<a name="method_getPlatformFeeAmount" class="anchor"></a>
#### public getPlatformFeeAmount() : \YooKassa\Model\AmountInterface|null

```php
public getPlatformFeeAmount() : \YooKassa\Model\AmountInterface|null
```

**Summary**

Возвращает комиссию за проданные товары и услуги, которая удерживается с магазина в вашу пользу.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\Source](../classes/YooKassa-Model-Refund-Source.md)

**Returns:** \YooKassa\Model\AmountInterface|null - Сумма комиссии


<a name="method_getValidator" class="anchor"></a>
#### public getValidator() : \YooKassa\Validator\Validator

```php
public getValidator() : \YooKassa\Validator\Validator
```

**Details:**
* Inherited From: [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)

**Returns:** \YooKassa\Validator\Validator - 


<a name="method_hasAmount" class="anchor"></a>
#### public hasAmount() : bool

```php
public hasAmount() : bool
```

**Summary**

Проверяет, была ли установлена сумма оплаты.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\Source](../classes/YooKassa-Model-Refund-Source.md)

**Returns:** bool - True если сумма оплаты была установлена, false если нет


<a name="method_hasPlatformFeeAmount" class="anchor"></a>
#### public hasPlatformFeeAmount() : bool

```php
public hasPlatformFeeAmount() : bool
```

**Summary**

Проверяет, была ли установлена комиссия за проданные товары и услуги, которая удерживается с магазина в вашу пользу.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\Source](../classes/YooKassa-Model-Refund-Source.md)

**Returns:** bool - True если комиссия была установлена, false если нет


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


<a name="method_setAccountId" class="anchor"></a>
#### public setAccountId() : self

```php
public setAccountId(string|null $account_id = null) : self
```

**Summary**

Устанавливает id магазина-получателя средств.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\Source](../classes/YooKassa-Model-Refund-Source.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | account_id  | Id магазина с которого будут списаны средства |

**Returns:** self - 


<a name="method_setAmount" class="anchor"></a>
#### public setAmount() : self

```php
public setAmount(\YooKassa\Model\AmountInterface|array|null $amount = null) : self
```

**Summary**

Устанавливает сумму оплаты.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\Source](../classes/YooKassa-Model-Refund-Source.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\AmountInterface OR array OR null</code> | amount  | Сумма оплаты |

**Returns:** self - 


<a name="method_setPlatformFeeAmount" class="anchor"></a>
#### public setPlatformFeeAmount() : self

```php
public setPlatformFeeAmount(\YooKassa\Model\AmountInterface|array|null $platform_fee_amount = null) : self
```

**Summary**

Устанавливает комиссию за проданные товары и услуги, которая удерживается с магазина в вашу пользу.

**Details:**
* Inherited From: [\YooKassa\Model\Refund\Source](../classes/YooKassa-Model-Refund-Source.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\AmountInterface OR array OR null</code> | platform_fee_amount  | Сумма комиссии |

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