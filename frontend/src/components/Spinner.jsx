import { DNA } from 'react-loader-spinner'

function Spinner({ size = 24, visible = true, ariaLabel = 'dna-loading', wrapperClass = 'dna-wrapper' }) {
  return (
    <DNA
      visible={visible}
      height={size}
      width={size}
      ariaLabel={ariaLabel}
      wrapperStyle={{ display: 'inline-block' }}
      wrapperClass={wrapperClass}
    />
  )
}

export default Spinner

/*
<Oval
      height={size}
      width={size}
      color={color}
      wrapperStyle={{ display: 'inline-block' }}
      wrapperClass=""
      visible={visible}
      ariaLabel={ariaLabel}
      secondaryColor={color}
      strokeWidth={2}
      strokeWidthSecondary={2}
    />
*/