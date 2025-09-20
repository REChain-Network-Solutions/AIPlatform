# [YooKassa API SDK](../home.md)

# Class: \YooKassa\Request\Payments\PaymentData\ElectronicCertificate\ElectronicCertificateArticle
### Namespace: [\YooKassa\Request\Payments\PaymentData\ElectronicCertificate](../namespaces/yookassa-request-payments-paymentdata-electroniccertificate.md)
---
**Summary:**

Класс, представляющий модель ElectronicCertificateArticle.

**Description:**

Товарная позиция в корзине покупки при оплате по электронному сертификату.

---
### Constants
* No constants found

---
### Properties
| Visibility | Name | Flag | Summary |
| ----------:| ---- | ---- | ------- |
| public | [$article_code](../classes/YooKassa-Request-Payments-PaymentData-ElectronicCertificate-ElectronicCertificateArticle.md#property_article_code) |  | Код товара в вашей системе. Максимум 128 символов. |
| public | [$article_name](../classes/YooKassa-Request-Payments-PaymentData-ElectronicCertificate-ElectronicCertificateArticle.md#property_article_name) |  | Название товара в вашей системе. Отображается на готовой платежной форме ЮKassa. Максимум 128 символов. |
| public | [$article_number](../classes/YooKassa-Request-Payments-PaymentData-ElectronicCertificate-ElectronicCertificateArticle.md#property_article_number) |  | Порядковый номер товара в корзине. От 1 до 999 включительно. |
| public | [$articleCode](../classes/YooKassa-Request-Payments-PaymentData-ElectronicCertificate-ElectronicCertificateArticle.md#property_articleCode) |  | Код товара в вашей системе. Максимум 128 символов. |
| public | [$articleName](../classes/YooKassa-Request-Payments-PaymentData-ElectronicCertificate-ElectronicCertificateArticle.md#property_articleName) |  | Название товара в вашей системе. Отображается на готовой платежной форме ЮKassa. Максимум 128 символов. |
| public | [$articleNumber](../classes/YooKassa-Request-Payments-PaymentData-ElectronicCertificate-ElectronicCertificateArticle.md#property_articleNumber) |  | Порядковый номер товара в корзине. От 1 до 999 включительно. |
| public | [$metadata](../classes/YooKassa-Request-Payments-PaymentData-ElectronicCertificate-ElectronicCertificateArticle.md#property_metadata) |  | Любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа). Передаются в виде набора пар «ключ-значение» и возвращаются в ответе от ЮKassa. Ограничения: максимум 16 ключей, имя ключа не больше 32 символов, значение ключа не больше 512 символов, тип данных — строка в формате UTF-8. |
| public | [$price](../classes/YooKassa-Request-Payments-PaymentData-ElectronicCertificate-ElectronicCertificateArticle.md#property_price) |  | Цена за единицу товара. |
| public | [$quantity](../classes/YooKassa-Request-Payments-PaymentData-ElectronicCertificate-ElectronicCertificateArticle.md#property_quantity) |  | Количество единиц товара. Формат: целое положительное число. |
| public | [$tru_code](../classes/YooKassa-Request-Payments-PaymentData-ElectronicCertificate-ElectronicCertificateArticle.md#property_tru_code) |  | Код ТРУ. 30 символов, две группы цифр, разделенные точкой. Формат: ~`NNNNNNNNN.NNNNNNNNNYYYYMMMMZZZ`, где ~`NNNNNNNNN.NNNNNNNNN` — код вида ТРУ по [Перечню ТРУ](https://esnsi.gosuslugi.ru/classifiers/10616/data?pg=1&p=1), ~`YYYY` — код производителя, ~`MMMM` — код модели, ~`ZZZ` — код страны производителя. Пример: ~`329921120.06001010200080001643`  [Как сформировать код ТРУ](https://yookassa.ru/developers/payment-acceptance/integration-scenarios/manual-integration/other/electronic-certificate/basics#payments-preparations-tru-code) |
| public | [$truCode](../classes/YooKassa-Request-Payments-PaymentData-ElectronicCertificate-ElectronicCertificateArticle.md#property_truCode) |  | Код ТРУ. 30 символов, две группы цифр, разделенные точкой. Формат: ~`NNNNNNNNN.NNNNNNNNNYYYYMMMMZZZ`, где ~`NNNNNNNNN.NNNNNNNNN` — код вида ТРУ по [Перечню ТРУ](https://esnsi.gosuslugi.ru/classifiers/10616/data?pg=1&p=1), ~`YYYY` — код производителя, ~`MMMM` — код модели, ~`ZZZ` — код страны производителя. Пример: ~`329921120.06001010200080001643`  [Как сформировать код ТРУ](https://yookassa.ru/developers/payment-acceptance/integration-scenarios/manual-integration/other/electronic-certificate/basics#payments-preparations-tru-code) |

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
| public | [getArticleCode()](../classes/YooKassa-Request-Payments-PaymentData-ElectronicCertificate-ElectronicCertificateArticle.md#method_getArticleCode) |  | Возвращает article_code. |
| public | [getArticleName()](../classes/YooKassa-Request-Payments-PaymentData-ElectronicCertificate-ElectronicCertificateArticle.md#method_getArticleName) |  | Возвращает article_name. |
| public | [getArticleNumber()](../classes/YooKassa-Request-Payments-PaymentData-ElectronicCertificate-ElectronicCertificateArticle.md#method_getArticleNumber) |  | Возвращает article_number. |
| public | [getMetadata()](../classes/YooKassa-Request-Payments-PaymentData-ElectronicCertificate-ElectronicCertificateArticle.md#method_getMetadata) |  | Возвращает metadata. |
| public | [getPrice()](../classes/YooKassa-Request-Payments-PaymentData-ElectronicCertificate-ElectronicCertificateArticle.md#method_getPrice) |  | Возвращает price. |
| public | [getQuantity()](../classes/YooKassa-Request-Payments-PaymentData-ElectronicCertificate-ElectronicCertificateArticle.md#method_getQuantity) |  | Возвращает quantity. |
| public | [getTruCode()](../classes/YooKassa-Request-Payments-PaymentData-ElectronicCertificate-ElectronicCertificateArticle.md#method_getTruCode) |  | Возвращает tru_code. |
| public | [getValidator()](../classes/YooKassa-Common-AbstractObject.md#method_getValidator) |  |  |
| public | [jsonSerialize()](../classes/YooKassa-Common-AbstractObject.md#method_jsonSerialize) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации. |
| public | [offsetExists()](../classes/YooKassa-Common-AbstractObject.md#method_offsetExists) |  | Проверяет наличие свойства. |
| public | [offsetGet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetGet) |  | Возвращает значение свойства. |
| public | [offsetSet()](../classes/YooKassa-Common-AbstractObject.md#method_offsetSet) |  | Устанавливает значение свойства. |
| public | [offsetUnset()](../classes/YooKassa-Common-AbstractObject.md#method_offsetUnset) |  | Удаляет свойство. |
| public | [setArticleCode()](../classes/YooKassa-Request-Payments-PaymentData-ElectronicCertificate-ElectronicCertificateArticle.md#method_setArticleCode) |  | Устанавливает article_code. |
| public | [setArticleName()](../classes/YooKassa-Request-Payments-PaymentData-ElectronicCertificate-ElectronicCertificateArticle.md#method_setArticleName) |  | Устанавливает article_name. |
| public | [setArticleNumber()](../classes/YooKassa-Request-Payments-PaymentData-ElectronicCertificate-ElectronicCertificateArticle.md#method_setArticleNumber) |  | Устанавливает article_number. |
| public | [setMetadata()](../classes/YooKassa-Request-Payments-PaymentData-ElectronicCertificate-ElectronicCertificateArticle.md#method_setMetadata) |  | Устанавливает metadata. |
| public | [setPrice()](../classes/YooKassa-Request-Payments-PaymentData-ElectronicCertificate-ElectronicCertificateArticle.md#method_setPrice) |  | Устанавливает price. |
| public | [setQuantity()](../classes/YooKassa-Request-Payments-PaymentData-ElectronicCertificate-ElectronicCertificateArticle.md#method_setQuantity) |  | Устанавливает quantity. |
| public | [setTruCode()](../classes/YooKassa-Request-Payments-PaymentData-ElectronicCertificate-ElectronicCertificateArticle.md#method_setTruCode) |  | Устанавливает tru_code. |
| public | [toArray()](../classes/YooKassa-Common-AbstractObject.md#method_toArray) |  | Возвращает ассоциативный массив со свойствами текущего объекта для его дальнейшей JSON сериализации Является алиасом метода AbstractObject::jsonSerialize(). |
| protected | [getUnknownProperties()](../classes/YooKassa-Common-AbstractObject.md#method_getUnknownProperties) |  | Возвращает массив свойств которые не существуют, но были заданы у объекта. |
| protected | [validatePropertyValue()](../classes/YooKassa-Common-AbstractObject.md#method_validatePropertyValue) |  |  |

---
### Details
* File: [lib/Request/Payments/PaymentData/ElectronicCertificate/ElectronicCertificateArticle.php](../../lib/Request/Payments/PaymentData/ElectronicCertificate/ElectronicCertificateArticle.php)
* Package: YooKassa\Model
* Class Hierarchy: 
  * [\YooKassa\Common\AbstractObject](../classes/YooKassa-Common-AbstractObject.md)
  * \YooKassa\Request\Payments\PaymentData\ElectronicCertificate\ElectronicCertificateArticle

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
<a name="property_article_code"></a>
#### public $article_code : string
---
***Description***

Код товара в вашей системе. Максимум 128 символов.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_article_name"></a>
#### public $article_name : string
---
***Description***

Название товара в вашей системе. Отображается на готовой платежной форме ЮKassa. Максимум 128 символов.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_article_number"></a>
#### public $article_number : int
---
***Description***

Порядковый номер товара в корзине. От 1 до 999 включительно.

**Type:** <a href="../int"><abbr title="int">int</abbr></a>

**Details:**


<a name="property_articleCode"></a>
#### public $articleCode : string
---
***Description***

Код товара в вашей системе. Максимум 128 символов.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_articleName"></a>
#### public $articleName : string
---
***Description***

Название товара в вашей системе. Отображается на готовой платежной форме ЮKassa. Максимум 128 символов.

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_articleNumber"></a>
#### public $articleNumber : int
---
***Description***

Порядковый номер товара в корзине. От 1 до 999 включительно.

**Type:** <a href="../int"><abbr title="int">int</abbr></a>

**Details:**


<a name="property_metadata"></a>
#### public $metadata : \YooKassa\Model\Metadata
---
***Description***

Любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа). Передаются в виде набора пар «ключ-значение» и возвращаются в ответе от ЮKassa. Ограничения: максимум 16 ключей, имя ключа не больше 32 символов, значение ключа не больше 512 символов, тип данных — строка в формате UTF-8.

**Type:** <a href="../classes/YooKassa-Model-Metadata.html"><abbr title="\YooKassa\Model\Metadata">Metadata</abbr></a>

**Details:**


<a name="property_price"></a>
#### public $price : \YooKassa\Model\AmountInterface
---
***Description***

Цена за единицу товара.

**Type:** <a href="../classes/YooKassa-Model-AmountInterface.html"><abbr title="\YooKassa\Model\AmountInterface">AmountInterface</abbr></a>

**Details:**


<a name="property_quantity"></a>
#### public $quantity : int
---
***Description***

Количество единиц товара. Формат: целое положительное число.

**Type:** <a href="../int"><abbr title="int">int</abbr></a>

**Details:**


<a name="property_tru_code"></a>
#### public $tru_code : string
---
***Description***

Код ТРУ. 30 символов, две группы цифр, разделенные точкой. Формат: ~`NNNNNNNNN.NNNNNNNNNYYYYMMMMZZZ`, где ~`NNNNNNNNN.NNNNNNNNN` — код вида ТРУ по [Перечню ТРУ](https://esnsi.gosuslugi.ru/classifiers/10616/data?pg=1&p=1), ~`YYYY` — код производителя, ~`MMMM` — код модели, ~`ZZZ` — код страны производителя. Пример: ~`329921120.06001010200080001643`  [Как сформировать код ТРУ](https://yookassa.ru/developers/payment-acceptance/integration-scenarios/manual-integration/other/electronic-certificate/basics#payments-preparations-tru-code)

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

**Details:**


<a name="property_truCode"></a>
#### public $truCode : string
---
***Description***

Код ТРУ. 30 символов, две группы цифр, разделенные точкой. Формат: ~`NNNNNNNNN.NNNNNNNNNYYYYMMMMZZZ`, где ~`NNNNNNNNN.NNNNNNNNN` — код вида ТРУ по [Перечню ТРУ](https://esnsi.gosuslugi.ru/classifiers/10616/data?pg=1&p=1), ~`YYYY` — код производителя, ~`MMMM` — код модели, ~`ZZZ` — код страны производителя. Пример: ~`329921120.06001010200080001643`  [Как сформировать код ТРУ](https://yookassa.ru/developers/payment-acceptance/integration-scenarios/manual-integration/other/electronic-certificate/basics#payments-preparations-tru-code)

**Type:** <a href="../string"><abbr title="string">string</abbr></a>

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


<a name="method_getArticleCode" class="anchor"></a>
#### public getArticleCode() : string|null

```php
public getArticleCode() : string|null
```

**Summary**

Возвращает article_code.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentData\ElectronicCertificate\ElectronicCertificateArticle](../classes/YooKassa-Request-Payments-PaymentData-ElectronicCertificate-ElectronicCertificateArticle.md)

**Returns:** string|null - Код товара в вашей системе


<a name="method_getArticleName" class="anchor"></a>
#### public getArticleName() : string|null

```php
public getArticleName() : string|null
```

**Summary**

Возвращает article_name.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentData\ElectronicCertificate\ElectronicCertificateArticle](../classes/YooKassa-Request-Payments-PaymentData-ElectronicCertificate-ElectronicCertificateArticle.md)

**Returns:** string|null - Название товара в вашей системе


<a name="method_getArticleNumber" class="anchor"></a>
#### public getArticleNumber() : int|null

```php
public getArticleNumber() : int|null
```

**Summary**

Возвращает article_number.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentData\ElectronicCertificate\ElectronicCertificateArticle](../classes/YooKassa-Request-Payments-PaymentData-ElectronicCertificate-ElectronicCertificateArticle.md)

**Returns:** int|null - Порядковый номер товара в корзине


<a name="method_getMetadata" class="anchor"></a>
#### public getMetadata() : \YooKassa\Model\Metadata|null

```php
public getMetadata() : \YooKassa\Model\Metadata|null
```

**Summary**

Возвращает metadata.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentData\ElectronicCertificate\ElectronicCertificateArticle](../classes/YooKassa-Request-Payments-PaymentData-ElectronicCertificate-ElectronicCertificateArticle.md)

**Returns:** \YooKassa\Model\Metadata|null - Любые дополнительные данные


<a name="method_getPrice" class="anchor"></a>
#### public getPrice() : \YooKassa\Model\AmountInterface|null

```php
public getPrice() : \YooKassa\Model\AmountInterface|null
```

**Summary**

Возвращает price.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentData\ElectronicCertificate\ElectronicCertificateArticle](../classes/YooKassa-Request-Payments-PaymentData-ElectronicCertificate-ElectronicCertificateArticle.md)

**Returns:** \YooKassa\Model\AmountInterface|null - Цена за единицу товара


<a name="method_getQuantity" class="anchor"></a>
#### public getQuantity() : int|null

```php
public getQuantity() : int|null
```

**Summary**

Возвращает quantity.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentData\ElectronicCertificate\ElectronicCertificateArticle](../classes/YooKassa-Request-Payments-PaymentData-ElectronicCertificate-ElectronicCertificateArticle.md)

**Returns:** int|null - Количество единиц товара


<a name="method_getTruCode" class="anchor"></a>
#### public getTruCode() : string|null

```php
public getTruCode() : string|null
```

**Summary**

Возвращает tru_code.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentData\ElectronicCertificate\ElectronicCertificateArticle](../classes/YooKassa-Request-Payments-PaymentData-ElectronicCertificate-ElectronicCertificateArticle.md)

**Returns:** string|null - Код ТРУ


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


<a name="method_setArticleCode" class="anchor"></a>
#### public setArticleCode() : self

```php
public setArticleCode(string|null $article_code = null) : self
```

**Summary**

Устанавливает article_code.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentData\ElectronicCertificate\ElectronicCertificateArticle](../classes/YooKassa-Request-Payments-PaymentData-ElectronicCertificate-ElectronicCertificateArticle.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | article_code  | Код товара в вашей системе. Максимум 128 символов. |

**Returns:** self - 


<a name="method_setArticleName" class="anchor"></a>
#### public setArticleName() : self

```php
public setArticleName(string|null $article_name = null) : self
```

**Summary**

Устанавливает article_name.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentData\ElectronicCertificate\ElectronicCertificateArticle](../classes/YooKassa-Request-Payments-PaymentData-ElectronicCertificate-ElectronicCertificateArticle.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | article_name  | Название товара в вашей системе. Отображается на готовой платежной форме ЮKassa. Максимум 128 символов. |

**Returns:** self - 


<a name="method_setArticleNumber" class="anchor"></a>
#### public setArticleNumber() : self

```php
public setArticleNumber(int|null $article_number = null) : self
```

**Summary**

Устанавливает article_number.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentData\ElectronicCertificate\ElectronicCertificateArticle](../classes/YooKassa-Request-Payments-PaymentData-ElectronicCertificate-ElectronicCertificateArticle.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">int OR null</code> | article_number  | Порядковый номер товара в корзине. От 1 до 999 включительно. |

**Returns:** self - 


<a name="method_setMetadata" class="anchor"></a>
#### public setMetadata() : self

```php
public setMetadata(\YooKassa\Model\Metadata|array|null $metadata = null) : self
```

**Summary**

Устанавливает metadata.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentData\ElectronicCertificate\ElectronicCertificateArticle](../classes/YooKassa-Request-Payments-PaymentData-ElectronicCertificate-ElectronicCertificateArticle.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\Metadata OR array OR null</code> | metadata  | Любые дополнительные данные, которые нужны вам для работы (например, ваш внутренний идентификатор заказа). Передаются в виде набора пар «ключ-значение» и возвращаются в ответе от ЮKassa. Ограничения: максимум 16 ключей, имя ключа не больше 32 символов, значение ключа не больше 512 символов, тип данных — строка в формате UTF-8. |

**Returns:** self - 


<a name="method_setPrice" class="anchor"></a>
#### public setPrice() : self

```php
public setPrice(\YooKassa\Model\AmountInterface|array|null $price = null) : self
```

**Summary**

Устанавливает price.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentData\ElectronicCertificate\ElectronicCertificateArticle](../classes/YooKassa-Request-Payments-PaymentData-ElectronicCertificate-ElectronicCertificateArticle.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">\YooKassa\Model\AmountInterface OR array OR null</code> | price  | Цена за единицу товара. |

**Returns:** self - 


<a name="method_setQuantity" class="anchor"></a>
#### public setQuantity() : self

```php
public setQuantity(int|null $quantity = null) : self
```

**Summary**

Устанавливает quantity.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentData\ElectronicCertificate\ElectronicCertificateArticle](../classes/YooKassa-Request-Payments-PaymentData-ElectronicCertificate-ElectronicCertificateArticle.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">int OR null</code> | quantity  | Количество единиц товара. Формат: целое положительное число. |

**Returns:** self - 


<a name="method_setTruCode" class="anchor"></a>
#### public setTruCode() : self

```php
public setTruCode(string|null $tru_code = null) : self
```

**Summary**

Устанавливает tru_code.

**Details:**
* Inherited From: [\YooKassa\Request\Payments\PaymentData\ElectronicCertificate\ElectronicCertificateArticle](../classes/YooKassa-Request-Payments-PaymentData-ElectronicCertificate-ElectronicCertificateArticle.md)

##### Parameters:
| Type | Name | Description |
| ---- | ---- | ----------- |
| <code lang="php">string OR null</code> | tru_code  | Код ТРУ. 30 символов |

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