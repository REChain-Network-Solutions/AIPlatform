<?php

/*
* The MIT License
*
* Copyright (c) 2025 "YooMoney", NBСO LLC
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*/

namespace YooKassa\Common\Errors;

use YooKassa\Common\AbstractObject;
use YooKassa\Validator\Constraints as Assert;

/**
 * Класс, представляющий модель AbstractError.
 *
 * Объект ошибки.
 *
 * @category Class
 * @package  YooKassa\Model
 * @author   cms@yoomoney.ru
 * @link     https://yookassa.ru/developers/api
 *
 * @property string $type Тип объекта.
 * @property string $id Идентификатор ошибки. Используйте его, если возникла необходимость обратиться в техническую поддержку.
 * @property string $description Подробное описание ошибки.
 * @property string $parameter Название параметра, из-за которого произошла ошибка.
 * @property int $retry_after Рекомендуемое количество миллисекунд, спустя которое следует повторить запрос.
 * @property int $retryAfter Рекомендуемое количество миллисекунд, спустя которое следует повторить запрос.
*/
abstract class AbstractError extends AbstractObject
{
    /**
     * Тип объекта.
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    private ?string $_type = null;

    /**
     * Идентификатор ошибки. Используйте его, если возникла необходимость обратиться в техническую поддержку.
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    private ?string $_id = null;

    /**
     * Код ошибки.
     *
     * @var string|null
     */
    #[Assert\NotBlank]
    #[Assert\Choice(callback: [ErrorCode::class, 'getValidValues'])]
    #[Assert\Type('string')]
    protected ?string $_code = null;

    /**
     * Подробное описание ошибки.
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    private ?string $_description = null;

    /**
     * Название параметра, из-за которого произошла ошибка.
     *
     * @var string|null
     */
    #[Assert\Type('string')]
    private ?string $_parameter = null;

    /**
     * Рекомендуемое количество миллисекунд, спустя которое следует повторить запрос.
     *
     * @var int|null
     */
    #[Assert\Type('int')]
    private ?int $_retry_after = null;

    /**
     * Возвращает type.
     *
     * @return string|null
     */
    public function getType(): ?string
    {
        return $this->_type;
    }

    /**
     * Устанавливает type.
     *
     * @param string|null $type Тип объекта.
     *
     * @return self
     */
    public function setType(?string $type = null): self
    {
        $this->_type = $this->validatePropertyValue('_type', $type);
        return $this;
    }

    /**
     * Возвращает id.
     *
     * @return string|null
     */
    public function getId(): ?string
    {
        return $this->_id;
    }

    /**
     * Устанавливает id.
     *
     * @param string|null $id Идентификатор ошибки. Используйте его, если возникла необходимость обратиться в техническую поддержку.
     *
     * @return self
     */
    public function setId(?string $id = null): self
    {
        $this->_id = $this->validatePropertyValue('_id', $id);
        return $this;
    }

    /**
     * Возвращает code.
     *
     * @return string|null
     */
    public function getCode(): ?string
    {
        return $this->_code;
    }

    /**
     * Устанавливает code.
     *
     * @param string|null $code Код ошибки.
     *
     * @return self
     */
    public function setCode(?string $code = null): self
    {
        $this->_code = $this->validatePropertyValue('_code', $code);
        return $this;
    }

    /**
     * Возвращает description.
     *
     * @return string|null
     */
    public function getDescription(): ?string
    {
        return $this->_description;
    }

    /**
     * Устанавливает description.
     *
     * @param string|null $description Подробное описание ошибки.
     *
     * @return self
     */
    public function setDescription(?string $description = null): self
    {
        $this->_description = $this->validatePropertyValue('_description', $description);
        return $this;
    }

    /**
     * Возвращает parameter.
     *
     * @return string|null
     */
    public function getParameter(): ?string
    {
        return $this->_parameter;
    }

    /**
     * Устанавливает parameter.
     *
     * @param string|null $parameter Название параметра, из-за которого произошла ошибка.
     *
     * @return self
     */
    public function setParameter(?string $parameter = null): self
    {
        $this->_parameter = $this->validatePropertyValue('_parameter', $parameter);
        return $this;
    }

    /**
     * Возвращает retry_after.
     *
     * @return int|null
     */
    public function getRetryAfter(): ?int
    {
        return $this->_retry_after;
    }

    /**
     * Устанавливает retry_after.
     *
     * @param int|null $retry_after Рекомендуемое количество миллисекунд, спустя которое следует повторить запрос.
     *
     * @return self
     */
    public function setRetryAfter(?int $retry_after = null): self
    {
        $this->_retry_after = $this->validatePropertyValue('_retry_after', $retry_after);
        return $this;
    }

}

